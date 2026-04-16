import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import time
import joblib

class HybridDiseaseModel:
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=5,          # Reduced from 8 — prevents overfitting deep trees
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=5,   # Higher value = more conservative splits
            gamma=0.4,            # Stronger minimum loss reduction to split
            reg_alpha=0.1,        # L1 regularization — sparsifies weights
            reg_lambda=1.5,       # L2 regularization — penalizes large weights
            eval_metric='mlogloss',
            random_state=42,
            tree_method='hist'
        )
        self.save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.save_path, exist_ok=True)
        self.classes = ['Tomato_Early_Blight', 'Tomato_Late_Blight', 'Tomato_Leaf_Mold', 'Tomato_healthy']

    def extract_features(self, generator):
        """Extract features from images using MobileNetV2."""
        print(f"Extracting features using MobileNetV2 for {generator.samples} images...")
        features = self.base_model.predict(generator, verbose=1)
        labels = generator.classes
        return features, labels

    def train_hybrid(self, data_dir, k_folds=5):
        """Train the hybrid model (MobileNetV2 + XGBoost) with K-Fold Cross Validation."""
        print(f"Training hybrid disease model with {k_folds}-fold cross-validation...")
        
        # Get all image paths and labels
        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
            data_dir, target_size=(224, 224), batch_size=32, class_mode='sparse', 
            shuffle=False, classes=self.classes
        )
        
        # Get file paths and labels
        filepaths = generator.filepaths
        labels = generator.classes
        
        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        accuracies = []
        f1_scores = []
        training_times = []
        
        fold = 1
        for train_idx, val_idx in skf.split(filepaths, labels):
            print(f"\n--- Fold {fold}/{k_folds} ---")
            
            # Create data generators for this fold
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=25,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.25,
                horizontal_flip=True,
                vertical_flip=False,
                brightness_range=[0.75, 1.25],
                channel_shift_range=20.0,
                fill_mode='reflect'
            )
            
            val_datagen = ImageDataGenerator(rescale=1./255)
            
            train_generator = train_datagen.flow_from_dataframe(
                pd.DataFrame({'filename': np.array(filepaths)[train_idx], 'class': np.array(labels)[train_idx]}),
                x_col='filename', y_col='class', target_size=(224, 224), batch_size=32, class_mode='sparse', shuffle=True
            )
            
            val_generator = val_datagen.flow_from_dataframe(
                pd.DataFrame({'filename': np.array(filepaths)[val_idx], 'class': np.array(labels)[val_idx]}),
                x_col='filename', y_col='class', target_size=(224, 224), batch_size=32, class_mode='sparse', shuffle=False
            )
            
            start_time = time.time()
            
            # Extract features for this fold
            X_train, y_train = self.extract_features(train_generator)
            X_val, y_val = self.extract_features(val_generator)
            
            # Normalize features using train stats only
            train_mean, train_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
            X_train = (X_train - train_mean) / train_std
            X_val = (X_val - train_mean) / train_std
            
            # Calculate class weights
            class_weights = {i: len(y_train) / (len(np.unique(y_train)) * np.sum(y_train == i)) for i in np.unique(y_train)}
            
            # Train XGBoost
            self.xgb_model.fit(
                X_train, y_train,
                sample_weight=[class_weights.get(y, 1) for y in y_train]
            )
            
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Evaluate
            y_pred = self.xgb_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            accuracies.append(accuracy)
            f1_scores.append(f1)
            
            print(f"Fold {fold} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
            fold += 1
        
        # Save the final model (trained on all data)
        print("\nTraining final model on all data...")
        all_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.25,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.75, 1.25],
            channel_shift_range=20.0,
            fill_mode='reflect'
        )
        
        final_generator = all_datagen.flow_from_directory(
            data_dir, target_size=(224, 224), batch_size=32, class_mode='sparse', 
            shuffle=True, classes=self.classes
        )
        
        X_all, y_all = self.extract_features(final_generator)
        train_mean, train_std = X_all.mean(axis=0), X_all.std(axis=0) + 1e-8
        X_all = (X_all - train_mean) / train_std
        
        class_weights = {i: len(y_all) / (len(np.unique(y_all)) * np.sum(y_all == i)) for i in np.unique(y_all)}
        
        self.xgb_model.fit(
            X_all, y_all,
            sample_weight=[class_weights.get(y, 1) for y in y_all]
        )
        
        joblib.dump(self.xgb_model, os.path.join(self.save_path, "disease_xgb.joblib"))
        
        return {
            "model": "MobileNetV2 + XGBoost",
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "training_time_mean": np.mean(training_times),
            "k_folds": k_folds
        }

    def train_baseline(self, data_dir, k_folds=5):
        """Train a pure MobileNetV2 model for comparison with K-Fold Cross Validation."""
        print(f"Training baseline MobileNetV2 model with {k_folds}-fold cross-validation...")
        
        # Get all image paths and labels
        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
            data_dir, target_size=(224, 224), batch_size=32, class_mode='sparse', 
            shuffle=False, classes=self.classes
        )
        
        filepaths = generator.filepaths
        labels = generator.classes
        
        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        accuracies = []
        training_times = []
        
        fold = 1
        for train_idx, val_idx in skf.split(filepaths, labels):
            print(f"\n--- Fold {fold}/{k_folds} ---")
            
            # Create data generators for this fold
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=25,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.3,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            val_datagen = ImageDataGenerator(rescale=1./255)
            
            train_generator = train_datagen.flow_from_dataframe(
                pd.DataFrame({'filename': np.array(filepaths)[train_idx], 'class': np.array(labels)[train_idx]}),
                x_col='filename', y_col='class', target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=True
            )
            
            val_generator = val_datagen.flow_from_dataframe(
                pd.DataFrame({'filename': np.array(filepaths)[val_idx], 'class': np.array(labels)[val_idx]}),
                x_col='filename', y_col='class', target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
            )
            
            # Build model for this fold
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
            
            # Unfreeze last few layers for fine-tuning
            for layer in base_model.layers[-20:]:
                layer.trainable = True
            
            x = layers.Dense(256, activation='relu')(base_model.output)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(self.num_classes, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=x)
            
            optimizer = Adam(learning_rate=1e-4)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            
            start_time = time.time()
            
            # Callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
            
            model.fit(
                train_generator,
                epochs=20,
                validation_data=val_generator,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            _, accuracy = model.evaluate(val_generator, verbose=0)
            accuracies.append(accuracy)
            
            print(f"Fold {fold} - Accuracy: {accuracy:.4f}, Time: {training_time:.2f}s")
            fold += 1
        
        # Train final model on all data
        print("\nTraining final baseline model on all data...")
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        final_generator = train_datagen.flow_from_directory(
            data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', classes=self.classes
        )
        
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        
        x = layers.Dense(256, activation='relu')(base_model.output)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.num_classes, activation='softmax')(x)
        final_model = Model(inputs=base_model.input, outputs=x)
        
        optimizer = Adam(learning_rate=1e-4)
        final_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        final_model.fit(
            final_generator,
            epochs=20,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        final_model.save(os.path.join(self.save_path, "disease_mobilenet.h5"))
        
        return {
            "model": "MobileNetV2 Alone",
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "training_time_mean": np.mean(training_times),
            "k_folds": k_folds
        }

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Tomato Dataset", "Tomato Diseases")
    
    hybrid = HybridDiseaseModel()
    h_results = hybrid.train_hybrid(data_dir)
    print("Hybrid Results:", h_results)
    
    b_results = hybrid.train_baseline(data_dir)
    print("Baseline Results:", b_results)
