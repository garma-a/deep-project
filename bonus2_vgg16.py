import os, time, warnings
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# 1. DOWNLOAD DATASET DIRECTLY
print("Downloading Brain Tumor MRI dataset directly from Kaggle...")
dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

def find_dir(name, path):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            if d.lower() == name.lower(): return os.path.join(root, d)
    return None

TRAIN_DIR = find_dir('training', dataset_path) or dataset_path
TEST_DIR = find_dir('testing', dataset_path) or dataset_path

# 2. CONFIGURATION
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 15 
LEARNING_RATE = 0.0001
SEED          = 42
OUTPUT_DIR    = './bonus_2_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. DATA GENERATORS
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    rotation_range=15,        
    zoom_range=0.1,           
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2      
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', seed=SEED, subset='training')
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', seed=SEED, subset='validation')
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False)

NUM_CLASSES = len(train_gen.class_indices)

def extract_features_from_generator(generator, feature_model):
    generator.reset()
    features, labels = [], []
    steps = len(generator)
    for i in range(steps):
        X_batch, y_batch = next(generator)
        feat = feature_model.predict(X_batch, verbose=0)
        features.append(feat)
        labels.append(y_batch)
    return np.vstack(features), np.vstack(labels)

# 4. APPROACH 1: SVM
print("\n=== APPROACH 1: VGG16 Features + SVM (Linear) ===")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
feature_extractor = models.Sequential([base_model, layers.GlobalAveragePooling2D()])

print("Extracting features...")
X_train_feat, y_train_raw = extract_features_from_generator(train_gen, feature_extractor)
X_val_feat,   y_val_raw   = extract_features_from_generator(val_gen, feature_extractor)
X_test_feat,  y_test_raw  = extract_features_from_generator(test_gen, feature_extractor)

y_train_int = np.argmax(y_train_raw, axis=1)
y_test_int  = np.argmax(y_test_raw,  axis=1)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_feat)
X_test_sc  = scaler.transform(X_test_feat)

svm_model = SVC(kernel='linear', C=1.0, random_state=SEED)
svm_model.fit(X_train_sc, y_train_int)
y_pred_svm = svm_model.predict(X_test_sc)

svm_acc  = accuracy_score(y_test_int, y_pred_svm)
svm_prec = precision_score(y_test_int, y_pred_svm, average='weighted', zero_division=0)
svm_rec  = recall_score(y_test_int, y_pred_svm, average='weighted', zero_division=0)
svm_f1   = f1_score(y_test_int, y_pred_svm, average='weighted', zero_division=0)
print(f"SVM Accuracy: {svm_acc:.4f}")

# 5. APPROACH 2: END-TO-END
print("\n=== APPROACH 2: End-to-End VGG16 (Fine-Tuning) ===")
train_gen.reset(); val_gen.reset(); test_gen.reset()

base_e2e = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_e2e.trainable = False
e2e_model = models.Sequential([
    base_e2e, layers.GlobalAveragePooling2D(), layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), layers.Dense(NUM_CLASSES, activation='softmax')
])

print("Phase 1: Warm-up")
e2e_model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
e2e_model.fit(train_gen, validation_data=val_gen, epochs=3, callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])

print("Phase 2: Fine-Tuning")
for layer in base_e2e.layers[-4:]: layer.trainable = True
e2e_model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE/10), loss='categorical_crossentropy', metrics=['accuracy'])
e2e_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

test_gen.reset()
y_pred_e2e_prob = e2e_model.predict(test_gen)
y_pred_e2e = np.argmax(y_pred_e2e_prob, axis=1)
y_true_e2e = test_gen.classes[:len(y_pred_e2e)]

e2e_acc  = accuracy_score(y_true_e2e, y_pred_e2e)
e2e_prec = precision_score(y_true_e2e, y_pred_e2e, average='weighted', zero_division=0)
e2e_rec  = recall_score(y_true_e2e, y_pred_e2e, average='weighted', zero_division=0)
e2e_f1   = f1_score(y_true_e2e, y_pred_e2e, average='weighted', zero_division=0)
print(f"E2E Accuracy: {e2e_acc:.4f}")

# 6. CHARTING
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
svm_scores = [svm_acc, svm_prec, svm_rec, svm_f1]
e2e_scores = [e2e_acc, e2e_prec, e2e_rec, e2e_f1]
x = np.arange(len(metrics)); width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, svm_scores, width, label='Approach 1 (VGG16 + SVM)', color='mediumpurple')
bars2 = ax.bar(x + width/2, e2e_scores, width, label='Approach 2 (End-to-End VGG16)', color='teal')

ax.set_ylabel('Score')
ax.set_title('Bonus Comparative Analysis: VGG16 + SVM vs End-to-End VGG16\n(Brain Tumor MRI Dataset)')
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0, 1.1)
ax.legend(loc='lower right'); ax.grid(axis='y', alpha=0.3)

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

chart_path = f'{OUTPUT_DIR}/bonus_vgg16_comparison_chart.png'
plt.savefig(chart_path, dpi=150)
print(f"\n✅ Chart saved successfully to: {chart_path}")
