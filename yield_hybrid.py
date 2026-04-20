import os
import numpy as np
import pandas as pd
import keras
from keras import layers, models
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
        
        l2_reg = keras.regularizers.l2(1e-4)

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
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={'latent': 'mse', 'reconstruction': 'mse'},
            loss_weights={'latent': 0, 'reconstruction': 1}
        )
        return model

    def train_hybrid(self, X, y, k_folds=5):
        """Train Hybrid Model: TabNet (Features) + XGBoost (Prediction) with K-Fold Cross Validation."""
        print(f"Training hybrid yield model with {k_folds}-fold cross-validation...")
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        training_times = []
        
        fold = 1
        for train_idx, val_idx in kf.split(X):
            print(f"\n--- Fold {fold}/{k_folds} ---")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            start_time = time.time()
            
            # Pre-train TabNet as autoencoder
            targets = {
                'latent': np.zeros((len(X_train_scaled), 128)),
                'reconstruction': X_train_scaled
            }
            self.tabnet.fit(
                X_train_scaled, targets,
                epochs=100,
                batch_size=16,
                verbose=0,
                validation_split=0.2,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
                ]
            )
            
            # Extract features
            preds = self.tabnet.predict(X_train_scaled)
            X_train_feat = preds['latent'] if isinstance(preds, dict) else preds[0]
            
            preds_val = self.tabnet.predict(X_val_scaled)
            X_val_feat = preds_val['latent'] if isinstance(preds_val, dict) else preds_val[0]
            
            # Train XGBoost
            self.xgb_model.fit(X_train_feat, y_train)
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Evaluate
            y_pred = self.xgb_model.predict(X_val_feat)
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            
            print(f"Fold {fold} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, Time: {training_time:.2f}s")
            fold += 1
        
        # Train final model on all data
        print("\nTraining final hybrid model on all data...")
        X_scaled = self.scaler.fit_transform(X)
        
        targets = {
            'latent': np.zeros((len(X_scaled), 128)),
            'reconstruction': X_scaled
        }
        self.tabnet.fit(
            X_scaled, targets,
            epochs=100,
            batch_size=16,
            verbose=0,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
        )
        
        preds = self.tabnet.predict(X_scaled)
        X_feat = preds['latent'] if isinstance(preds, dict) else preds[0]
        
        self.xgb_model.fit(X_feat, y)
        
        # Save models
        joblib.dump(self.xgb_model, os.path.join(self.save_path, "yield_xgb.joblib"))
        joblib.dump(self.scaler, os.path.join(self.save_path, "yield_scaler.joblib"))
        self.tabnet.save(os.path.join(self.save_path, "yield_tabnet_feat.h5"))
        
        return {
            "model": "TabNet + XGBoost",
            "r2_mean": np.mean(r2_scores),
            "r2_std": np.std(r2_scores),
            "rmse_mean": np.mean(rmse_scores),
            "rmse_std": np.std(rmse_scores),
            "mae_mean": np.mean(mae_scores),
            "mae_std": np.std(mae_scores),
            "training_time_mean": np.mean(training_times),
            "k_folds": k_folds
        }

    def train_baseline(self, X, y, k_folds=5):
        """Train standalone TabNet for comparison with K-Fold Cross Validation."""
        print(f"Training baseline yield model with {k_folds}-fold cross-validation...")
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        r2_scores = []
        training_times = []
        
        fold = 1
        for train_idx, val_idx in kf.split(X):
            print(f"\n--- Fold {fold}/{k_folds} ---")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
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
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
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
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
                ]
            )
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            y_pred = model.predict(X_val_scaled, verbose=0)
            r2 = r2_score(y_val, y_pred)
            r2_scores.append(r2)
            
            print(f"Fold {fold} - R²: {r2:.4f}, Time: {training_time:.2f}s")
            fold += 1
        
        # Train final model on all data
        print("\nTraining final baseline model on all data...")
        X_scaled = self.scaler.fit_transform(X)
        
        latent_output = self.tabnet.output['latent']
        x = layers.Dense(32, activation='relu')(latent_output)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1)(x)
        final_model = models.Model(inputs=self.tabnet.input, outputs=out)
        final_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        final_model.fit(
            X_scaled, y,
            epochs=150,
            batch_size=16,
            verbose=0,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
        )
        
        return {
            "model": "TabNet Alone",
            "r2_mean": np.mean(r2_scores),
            "r2_std": np.std(r2_scores),
            "training_time_mean": np.mean(training_times),
            "k_folds": k_folds
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
