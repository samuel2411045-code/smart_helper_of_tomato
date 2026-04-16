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

    def train_hybrid(self, X, y_type, k_folds=5):
        """Train Hybrid Model: RF (Feature Learning) + Gradient Boosting (Classification) with K-Fold Cross Validation."""
        print(f"Training hybrid soil model with {k_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        accuracies = []
        balanced_accs = []
        f1_scores = []
        training_times = []
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y_type):
            print(f"\n--- Fold {fold}/{k_folds} ---")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_type[train_idx], y_type[val_idx]
            
            # Scale features for GB (RF is scale-invariant)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            start_time = time.time()
            
            # Train Random Forest
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            self.rf_model.fit(X_train, y_train)
            
            # Create stacked features
            rf_pred_train = self.rf_model.predict(X_train_scaled)
            rf_proba_train = self.rf_model.predict_proba(X_train_scaled)
            X_train_ext = np.column_stack((X_train_scaled, rf_proba_train))
            
            rf_pred_val = self.rf_model.predict(X_val_scaled)
            rf_proba_val = self.rf_model.predict_proba(X_val_scaled)
            X_val_ext = np.column_stack((X_val_scaled, rf_proba_val))
            
            # Train Gradient Boosting
            self.gb_model.fit(X_train_ext, y_train)
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Evaluate
            y_pred = self.gb_model.predict(X_val_ext)
            accuracy = accuracy_score(y_val, y_pred)
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            
            accuracies.append(accuracy)
            balanced_accs.append(balanced_acc)
            f1_scores.append(f1)
            
            print(f"Fold {fold} - Acc: {accuracy:.4f}, Bal Acc: {balanced_acc:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
            fold += 1
        
        # Train final model on all data
        print("\nTraining final hybrid model on all data...")
        X_scaled = self.scaler.fit_transform(X)
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_type), y=y_type)
        self.rf_model.fit(X, y_type)
        
        rf_pred_all = self.rf_model.predict(X_scaled)
        rf_proba_all = self.rf_model.predict_proba(X_scaled)
        X_ext = np.column_stack((X_scaled, rf_proba_all))
        
        self.gb_model.fit(X_ext, y_type)
        
        # Save models
        joblib.dump(self.rf_model, os.path.join(self.save_path, "soil_rf.joblib"))
        joblib.dump(self.gb_model, os.path.join(self.save_path, "soil_gb.joblib"))
        joblib.dump(self.scaler, os.path.join(self.save_path, "soil_scaler.joblib"))
        
        return {
            "model": "RF + Gradient Boosting",
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "balanced_accuracy_mean": np.mean(balanced_accs),
            "balanced_accuracy_std": np.std(balanced_accs),
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "training_time_mean": np.mean(training_times),
            "k_folds": k_folds
        }

    def train_baseline(self, X, y, k_folds=5):
        """Train standalone RF for comparison with K-Fold Cross Validation."""
        print(f"Training baseline soil model with {k_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        accuracies = []
        balanced_accs = []
        training_times = []
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            print(f"\n--- Fold {fold}/{k_folds} ---")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            start_time = time.time()
            self.rf_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            y_pred = self.rf_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            
            accuracies.append(accuracy)
            balanced_accs.append(balanced_acc)
            
            print(f"Fold {fold} - Acc: {accuracy:.4f}, Bal Acc: {balanced_acc:.4f}, Time: {training_time:.2f}s")
            fold += 1
        
        # Train final model on all data
        print("\nTraining final baseline model on all data...")
        self.rf_model.fit(X, y)
        
        return {
            "model": "Random Forest Alone",
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "balanced_accuracy_mean": np.mean(balanced_accs),
            "balanced_accuracy_std": np.std(balanced_accs),
            "training_time_mean": np.mean(training_times),
            "k_folds": k_folds
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
