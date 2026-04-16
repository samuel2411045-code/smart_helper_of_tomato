import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
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

    def train_hybrid(self, data_dir):
        """Train the hybrid model (MobileNetV2 + XGBoost)."""
        # Enhanced data augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.25,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.75, 1.25],  # Robust to lighting changes
            channel_shift_range=20.0,       # Color channel robustness
            fill_mode='reflect'             # Better edge handling than 'nearest'
        )
        
        train_gen = datagen.flow_from_directory(
            data_dir, target_size=(224, 224), batch_size=32, class_mode='sparse', 
            shuffle=True, subset='training', classes=self.classes)
        
        val_gen = datagen.flow_from_directory(
            data_dir, target_size=(224, 224), batch_size=32, class_mode='sparse', 
            shuffle=False, subset='validation', classes=self.classes)
        
        start_time = time.time()
        X_train, y_train = self.extract_features(train_gen)
        X_val, y_val = self.extract_features(val_gen)
        
        # Normalize features using train stats only (prevent data leakage)
        train_mean, train_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std
        
        # Unfreeze last few layers of MobileNetV2 for feature fine-tuning
        for layer in self.base_model.layers[-30:]:
            layer.trainable = True
        
        print("Training XGBoost Classifier with optimized parameters...")
        # Calculate class weights to handle imbalance
        class_weights = {i: len(y_train) / (len(np.unique(y_train)) * np.sum(y_train == i)) for i in np.unique(y_train)}
        
        self.xgb_model.fit(
            X_train, y_train,
            sample_weight=[class_weights.get(y, 1) for y in y_train]
        )
        training_time = time.time() - start_time
        
        y_pred = self.xgb_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # Save models
        joblib.dump(self.xgb_model, os.path.join(self.save_path, "disease_xgb.joblib"))
        
        return {
            "model": "MobileNetV2 + XGBoost",
            "accuracy": accuracy,
            "f1_score": f1,
            "training_time_sec": training_time,
            "inference_speed_ms": (training_time / len(X_val)) * 1000
        }

    def train_baseline(self, data_dir):
        """Train a pure MobileNetV2 model for comparison."""
        print("Training Enhanced Baseline MobileNetV2 with data augmentation...")
        
        # Enhanced augmentation for baseline
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
        
        train_gen = train_datagen.flow_from_directory(
            data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', classes=self.classes)
        
        val_gen = val_datagen.flow_from_directory(
            data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', classes=self.classes)
        
        # Unfreeze last few layers for fine-tuning
        for layer in self.base_model.layers[-20:]:
            layer.trainable = True
        
        x = layers.Dense(256, activation='relu')(self.base_model.output)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=self.base_model.input, outputs=x)
        
        # Compile with better optimizer settings
        optimizer = Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        start_time = time.time()
        
        # Callbacks for better training
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
        
        model.fit(
            train_gen,
            epochs=20,
            validation_data=val_gen,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        training_time = time.time() - start_time
        
        _, accuracy = model.evaluate(val_gen, verbose=0)
        
        # Save model
        model.save(os.path.join(self.save_path, "disease_mobilenet.h5"))
        
        return {
            "model": "MobileNetV2 Alone",
            "accuracy": accuracy,
            "training_time_sec": training_time
        }

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Tomato Dataset", "Tomato Diseases")
    
    hybrid = HybridDiseaseModel()
    h_results = hybrid.train_hybrid(data_dir)
    print("Hybrid Results:", h_results)
    
    b_results = hybrid.train_baseline(data_dir)
    print("Baseline Results:", b_results)
