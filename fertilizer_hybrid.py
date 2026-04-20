import os
import joblib
import pandas as pd
import numpy as np
from data_manager import DataManager
# from yield_hybrid import HybridYieldModel # Reusing TabNet logic - commented out for inference only
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import time
# import tensorflow as tf
# from tensorflow.keras import layers, models

class FertilizerRecommender:
    def __init__(self, input_dim=10): # N, P, K, temp, hum, pH, rainfall + 3 NPK ratios
        self.save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        # self.hybrid_yield_wrapper = HybridYieldModel(input_dim=input_dim)  # Commented out for inference
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        
    def train_recommender(self, k_folds=5):
        """Trains the fertilizer recommendation model using TabNet + XGBoost hybrid with K-Fold Cross Validation."""
        print(f"Training fertilizer recommender with {k_folds}-fold cross-validation...")
        
        dm = DataManager()
        df = dm.load_fertilizer_data()
        
        # Features: n, p, k, temperature, humidity, ph, rainfall
        features = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Feature Engineering: NPK Ratios
        df['n_p_ratio'] = df['n'] / (df['p'] + 1e-5)
        df['n_k_ratio'] = df['n'] / (df['k'] + 1e-5)
        df['p_k_ratio'] = df['p'] / (df['k'] + 1e-5)
        
        extended_features = features + ['n_p_ratio', 'n_k_ratio', 'p_k_ratio']
        X = df[extended_features].values
        y = df['label'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_enc = self.le.fit_transform(y)
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        accuracies = []
        f1_scores = []
        balanced_accs = []
        training_times = []
        
        fold = 1
        for train_idx, val_idx in skf.split(X_scaled, y_enc):
            print(f"\n--- Fold {fold}/{k_folds} ---")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_enc[train_idx], y_enc[val_idx]
            
            # Compute class weights
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            
            start_time = time.time()
            
            # Train TabNet Feature Learner
            targets = {
                'latent': np.zeros((len(X_train), 128)),
                'reconstruction': X_train
            }
            self.hybrid_yield_wrapper.tabnet.fit(
                X_train, targets,
                epochs=100,
                batch_size=16,
                verbose=0,
                validation_split=0.2,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=10, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
                    )
                ]
            )
            
            preds = self.hybrid_yield_wrapper.tabnet.predict(X_train)
            X_train_feat = preds['latent'] if isinstance(preds, dict) else preds[0]
            
            preds_val = self.hybrid_yield_wrapper.tabnet.predict(X_val)
            X_val_feat = preds_val['latent'] if isinstance(preds_val, dict) else preds_val[0]
            
            # Train XGBoost Classifier
            clf = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=4,
                subsample=0.80,
                colsample_bytree=0.80,
                min_child_weight=5,
                gamma=0.4,
                reg_alpha=0.1,
                reg_lambda=1.5,
                eval_metric='mlogloss',
                random_state=42,
                tree_method='hist'
            )
            
            clf.fit(
                X_train_feat, y_train,
                sample_weight=[class_weights[int(y)] for y in y_train]
            )
            
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Evaluate
            y_pred = clf.predict(X_val_feat)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            
            accuracies.append(accuracy)
            f1_scores.append(f1)
            balanced_accs.append(balanced_acc)
            
            print(f"Fold {fold} - Acc: {accuracy:.4f}, F1: {f1:.4f}, Bal Acc: {balanced_acc:.4f}, Time: {training_time:.2f}s")
            fold += 1
        
        # Train final model on all data
        print("\nTraining final fertilizer model on all data...")
        class_weights = compute_class_weight('balanced', classes=np.unique(y_enc), y=y_enc)
        
        targets = {
            'latent': np.zeros((len(X_scaled), 128)),
            'reconstruction': X_scaled
        }
        self.hybrid_yield_wrapper.tabnet.fit(
            X_scaled, targets,
            epochs=100,
            batch_size=16,
            verbose=0,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
                )
            ]
        )
        
        preds = self.hybrid_yield_wrapper.tabnet.predict(X_scaled)
        X_feat = preds['latent'] if isinstance(preds, dict) else preds[0]
        
        final_clf = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=5,
            gamma=0.4,
            reg_alpha=0.1,
            reg_lambda=1.5,
            eval_metric='mlogloss',
            random_state=42,
            tree_method='hist'
        )
        
        final_clf.fit(
            X_feat, y_enc,
            sample_weight=[class_weights[int(y)] for y in y_enc]
        )
        
        # Save models
        joblib.dump(final_clf, os.path.join(self.save_path, "fert_xgb.joblib"))
        joblib.dump(self.scaler, os.path.join(self.save_path, "fert_scaler.joblib"))
        joblib.dump(self.le, os.path.join(self.save_path, "fert_le.joblib"))
        
        return {
            "model": "TabNet + XGBoost",
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "balanced_accuracy_mean": np.mean(balanced_accs),
            "balanced_accuracy_std": np.std(balanced_accs),
            "training_time_mean": np.mean(training_times),
            "k_folds": k_folds
        }

    def get_recommendation(self, n, p, k, ph, rainfall, temperature, humidity, soil_type="Loamy", stage="Vegetative"):
        """Returns a detailed recommendation based on inputs."""
        # Nutrient deficiency analysis with thresholds
        deficiencies = []
        severity = []
        
        # Nitrogen analysis
        if n < 40:
            deficiencies.append("Nitrogen")
            severity.append("Critical" if n < 20 else "High")
        elif n < 100:
            deficiencies.append("Nitrogen")
            severity.append("Moderate")
        
        # Phosphorus analysis
        if p < 20:
            deficiencies.append("Phosphorus")
            severity.append("Critical" if p < 10 else "High")
        elif p < 40:
            deficiencies.append("Phosphorus")
            severity.append("Moderate")
        
        # Potassium analysis
        if k < 40:
            deficiencies.append("Potassium")
            severity.append("Critical" if k < 20 else "High")
        elif k < 100:
            deficiencies.append("Potassium")
            severity.append("Moderate")
        
        # Build deficiency string
        if deficiencies:
            def_str = ", ".join([f"{d} ({s})" for d, s in zip(deficiencies, severity)])
        else:
            def_str = "None - Optimal Nutrient Levels"
        
        # Quantity logic based on multiple factors
        base_qty = 50  # kg/ha
        
        # Stage multiplier
        stage_multiplier = {
            "Seedling": 0.5,
            "Early Vegetative": 0.8,
            "Vegetative": 1.0,
            "Flowering Initiation": 1.2,
            "Flowering": 1.3,
            "Fruiting": 1.5,
            "Unripe": 1.4,
            "Ripe": 1.0
        }
        base_qty *= stage_multiplier.get(stage, 1.0)
        
        # Soil type factor
        soil_factor = {
            "Clay": 1.0,
            "Sandy": 1.2,  # Needs more due to leaching
            "Loamy": 1.0,
            "Peaty": 0.8
        }
        base_qty *= soil_factor.get(soil_type, 1.0)
        
        # Rainfall factor (more rain = more leaching)
        if rainfall > 1500:
            base_qty *= 1.15
        elif rainfall < 500:
            base_qty *= 0.85
        
        # Temperature factor
        if temperature > 30:  # Hot climate
            base_qty *= 1.1
        elif temperature < 15:  # Cool climate
            base_qty *= 0.9
        
        # Fertilizer recommendation based on soil type
        recs = {
            "Clay": {"type": "Urea + DAP + MOP", "ratio": "2:1:1"},
            "Sandy": {"type": "Organic Compost + NPK", "ratio": "1:1:1.5"},
            "Loamy": {"type": "Balanced NPK 19-19-19", "ratio": "1:1:1"},
            "Peaty": {"type": "Balanced NPK 10-10-10", "ratio": "1:1:1"}
        }
        
        fert_info = recs.get(soil_type, {"type": "Standard NPK 19-19-19", "ratio": "1:1:1"})
        
        return {
            "recommended_fertilizer": fert_info["type"],
            "npk_ratio": fert_info["ratio"],
            "required_quantity_kg_ha": float(f"{base_qty:.2f}"),
            "nutrient_deficiency": def_str,
            "crop_stage": stage,
            "soil_type": soil_type,
            "application_notes": f"Apply in 2-3 splits for {stage} stage. Increase frequency in high rainfall areas."
        }

if __name__ == "__main__":
    fr = FertilizerRecommender()
    train_res = fr.train_recommender()
    print("Fertilizer Training Results:", train_res)
    
    # Test recommendation
    print("Sample Recommendation:", fr.get_recommendation(40, 20, 45, 6.5, 120, 26, 60, "Loamy", "Flowering"))
