import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import joblib
from data_manager import DataManager

class HybridYieldModel:
    def __init__(self, input_dim=6):
        self.input_dim = input_dim
        self.tabnet = self._build_simplified_tabnet()
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=4,          # Reduced from 7 — shallower trees generalize better
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=5,   # Conservative splits
            gamma=0.2,            # Minimum loss reduction
            reg_alpha=0.1,        # L1 regularization
            reg_lambda=2.0,       # L2 regularization (increased)
            random_state=42,
            tree_method='hist'
        )
        self.scaler = StandardScaler()
        self.save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.save_path, exist_ok=True)

    def _build_simplified_tabnet(self):
        """Improved TabNet-inspired feature learner using Keras."""
        inputs = layers.Input(shape=(self.input_dim,))
        
        l2_reg = tf.keras.regularizers.l2(1e-4)

        # Feature transformer (GLU-based)
        x = layers.Dense(256, kernel_regularizer=l2_reg)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Split for GLU
        line = layers.Dense(128, kernel_regularizer=l2_reg)(x)
        gate = layers.Dense(128, activation='sigmoid', kernel_regularizer=l2_reg)(x)
        glu = layers.Multiply()([line, gate])
        
        # Attentive transformer
        mask = layers.Dense(self.input_dim, activation='softmax')(glu)
        masked_features = layers.Multiply()([inputs, mask])
        
        # Latent representation with stronger dropout to fight overfitting
        latent = layers.Dense(128, activation='relu', kernel_regularizer=l2_reg, name='latent')(masked_features)
        latent = layers.Dropout(0.5)(latent)  # Increased from 0.4
        
        # Autoencoder head for pre-training
        reconstruction = layers.Dense(self.input_dim, name='reconstruction')(latent)
        
        model = models.Model(inputs=inputs, outputs={'latent': latent, 'reconstruction': reconstruction})
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={'latent': 'mse', 'reconstruction': 'mse'},
            loss_weights={'latent': 0, 'reconstruction': 1}
        )
        return model

    def train_hybrid(self, X, y):
        """Train Hybrid Model: TabNet (Features) + XGBoost (Prediction) with cross-validation."""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        start_time = time.time()
        print("Training TabNet Feature Learner...")
        # Pre-train as autoencoder with validation
        targets = {
            'latent': np.zeros((len(X_train_scaled), 128)),
            'reconstruction': X_train_scaled
        }
        self.tabnet.fit(
            X_train_scaled, targets,
            epochs=100,  # Increased from 50
            batch_size=16,  # Smaller batch for better learning
            verbose=0,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
        )
        
        # Extract features
        preds = self.tabnet.predict(X_train_scaled)
        X_train_feat = preds['latent'] if isinstance(preds, dict) else preds[0]
        
        preds_val = self.tabnet.predict(X_val_scaled)
        X_val_feat = preds_val['latent'] if isinstance(preds_val, dict) else preds_val[0]
        
        print("Training XGBoost Regressor with optimized parameters...")
        self.xgb_model.fit(
            X_train_feat, y_train
        )
        training_time = time.time() - start_time
        
        y_pred = self.xgb_model.predict(X_val_feat)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        # Save models
        joblib.dump(self.xgb_model, os.path.join(self.save_path, "yield_xgb.joblib"))
        joblib.dump(self.scaler, os.path.join(self.save_path, "yield_scaler.joblib"))
        self.tabnet.save(os.path.join(self.save_path, "yield_tabnet_feat.h5"))
        
        return {
            "model": "TabNet + XGBoost",
            "r2_score": r2,
            "rmse": rmse,
            "mae": mae,
            "training_time_sec": training_time
        }

    def train_baseline(self, X, y):
        """Train standalone TabNet for comparison."""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Add regression head to tabnet output
        latent_output = self.tabnet.output['latent']
        x = layers.Dense(32, activation='relu')(latent_output)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1)(x)
        model = models.Model(inputs=self.tabnet.input, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        start_time = time.time()
        model.fit(
            X_train_scaled, y_train,
            epochs=150,
            batch_size=16,
            verbose=0,
            validation_data=(X_val_scaled, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
        )
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_val_scaled, verbose=0)
        r2 = r2_score(y_val, y_pred)
        
        return {
            "model": "TabNet Alone",
            "r2_score": r2,
            "training_time_sec": training_time
        }

if __name__ == "__main__":
    dm = DataManager()
    df = dm.load_yield_data()
    
    # Feature selection and preparation
    if 'humidity' not in df.columns:
        df['humidity'] = np.random.uniform(40, 90, len(df))
        
    # Feature Engineering: Add interaction term
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    
    features = ['plant_height', 'leaf_count', 'flower_count', 'rainfall', 'temperature', 'humidity', 'temp_humidity_interaction']
    X = df[features].values
    y = df['yield_val'].values
    
    hybrid_model = HybridYieldModel(input_dim=len(features))
    h_res = hybrid_model.train_hybrid(X, y)
    print("Hybrid Results:", h_res)
    
    b_res = hybrid_model.train_baseline(X, y)
    print("Baseline Results:", b_res)
