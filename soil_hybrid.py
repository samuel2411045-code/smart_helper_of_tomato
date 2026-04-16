import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import time
import joblib
from data_manager import DataManager

class HybridSoilModel:
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,         # Reduced from 20 — prevents memorizing noise
            min_samples_split=6,  # Require more samples to split a node
            min_samples_leaf=5,   # Increased from 1 — smooths leaf predictions
            bootstrap=True,
            class_weight='balanced',
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            criterion='gini'
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,   # Slightly faster learning for shallower trees
            max_depth=4,          # Reduced from 6
            min_samples_split=8,  # More conservative splits
            min_samples_leaf=5,   # Increased to prevent leaf overfitting
            subsample=0.80,
            max_features='sqrt',
            random_state=42,
            validation_fraction=0.15,  # Larger validation fraction for more reliable early stopping
            n_iter_no_change=20   # Increased patience
        )
        
        self.scaler = StandardScaler()
        self.save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.save_path, exist_ok=True)

    def train_hybrid(self, X, y_type):
        """Train Hybrid Model: RF (Feature Learning) + Gradient Boosting (Classification) with CV."""
        X_train, X_val, y_train, y_val = train_test_split(X, y_type, test_size=0.2, random_state=42, stratify=y_type)
        
        # Scale features for GB (RF is scale-invariant)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        start_time = time.time()
        print("Training Random Forest Feature Learner...")
        
        # Compute class weights for RF
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        
        self.rf_model.fit(X_train, y_train)
        
        # Use RF predictions and probabilities as additional features for GB (Stacking)
        print("Creating stacked features using RF predictions...")
        rf_pred_train = self.rf_model.predict(X_train_scaled)
        rf_proba_train = self.rf_model.predict_proba(X_train_scaled)
        X_train_ext = np.column_stack((X_train_scaled, rf_proba_train))
        
        rf_pred_val = self.rf_model.predict(X_val_scaled)
        rf_proba_val = self.rf_model.predict_proba(X_val_scaled)
        X_val_ext = np.column_stack((X_val_scaled, rf_proba_val))
        
        print("Training Gradient Boosting Classifier...")
        self.gb_model.fit(X_train_ext, y_train)
        training_time = time.time() - start_time
        
        y_pred = self.gb_model.predict(X_val_ext)
        accuracy = accuracy_score(y_val, y_pred)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        # Save models
        joblib.dump(self.rf_model, os.path.join(self.save_path, "soil_rf.joblib"))
        joblib.dump(self.gb_model, os.path.join(self.save_path, "soil_gb.joblib"))
        joblib.dump(self.scaler, os.path.join(self.save_path, "soil_scaler.joblib"))
        
        return {
            "model": "RF + Gradient Boosting",
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "f1_score": f1,
            "training_time_sec": training_time
        }

    def train_baseline(self, X, y):
        """Train standalone RF for comparison with optimization."""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        start_time = time.time()
        self.rf_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = self.rf_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        
        return {
            "model": "Random Forest Alone",
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "training_time_sec": training_time
        }

if __name__ == "__main__":
    dm = DataManager()
    df = dm.load_soil_data()
    
    # Feature selection: n, p, k, temperature, humidity, ph, rainfall
    features = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[features].values
    y_type = df['soil_type'].values
    
    hybrid = HybridSoilModel()
    
    print("\n--- Soil Texture Prediction ---")
    h_res = hybrid.train_hybrid(X, y_type)
    print("Hybrid Results:", h_res)
    
    b_res = hybrid.train_baseline(X, y_type)
    print("Baseline Results:", b_res)
    
    # Moisture prediction (Similar hybrid approach)
    print("\n--- Soil Moisture Prediction ---")
    y_moist = df['moisture_type'].values
    m_h_res = hybrid.train_hybrid(X, y_moist)
    print("Moisture Hybrid Results:", m_h_res)
    
    # Save a separate moisture model for convenience
    joblib.dump(hybrid.gb_model, os.path.join(hybrid.save_path, "soil_moisture_gb.joblib"))
