
# ============================================================
# CELL 1: Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ============================================================
# CELL 2: Load Dataset from Google Drive
# ============================================================
# Since you already uploaded your zip to Google Drive:
#   1. Cell 1 must have run first (Drive is mounted)
#   2. Find your zip path in Drive: open the Files panel on the left,
#      navigate to MyDrive and copy the path of your zip file.
#   3. Paste it below as DRIVE_ZIP_PATH
# ============================================================
import zipfile, os

# ⬇️  This is your exact file path based on what you uploaded
DRIVE_ZIP_PATH = '/content/drive/MyDrive/deep-learning/archive.zip'

EXTRACT_PATH = '/content/brain_tumor_dataset'
os.makedirs(EXTRACT_PATH, exist_ok=True)

print(f"Extracting {DRIVE_ZIP_PATH} ...")
with zipfile.ZipFile(DRIVE_ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_PATH)

print("Done! Contents:")
print(os.listdir(EXTRACT_PATH))

# ============================================================
# CELL 3: Imports & Config
# ============================================================
import os, time, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# ---- CONFIGURATION ----
# Update DATASET_PATH to match your extracted folder structure.
# The folder should contain subfolders, one per class.
# e.g.  /content/brain_tumor_dataset/yes/  and  /content/brain_tumor_dataset/no/
# OR    /content/brain_tumor_dataset/Training/  and  /content/brain_tumor_dataset/Testing/
DATASET_PATH = '/content/brain_tumor_dataset'
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 25
LEARNING_RATE= 0.0001
SEED         = 42

# ---- OUTPUT FOLDER (saves all charts to Drive — survives session resets) ----
OUTPUT_DIR = '/content/drive/MyDrive/deep-learning/project_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Charts will be saved to: {OUTPUT_DIR}")

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ============================================================
# CELL 4: Explore Dataset Structure
# ============================================================
for root, dirs, files_ in os.walk(DATASET_PATH):
    level = root.replace(DATASET_PATH, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 2:
        subindent = ' ' * 2 * (level + 1)
        for f in files_[:3]:
            print(f'{subindent}{f}')

# ============================================================
# CELL 5: Data Generators (Preprocessing + Augmentation)
# ============================================================
# Training augmentation (justified: MRI datasets are small, augmentation
# simulates different patient positions inside the scanner)
# NOTE: preprocess_input applies ResNet50's exact ImageNet channel-wise
# mean subtraction & scaling — must match how the model was originally trained.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # ResNet50-specific preprocessing
    rotation_range=15,        # Slight rotation (scanner tilt)
    horizontal_flip=False,    # DISABLED: left/right orientation matters in MRI
    zoom_range=0.1,           # Minor zoom variation
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2      # 80% train, 20% validation
)

# No augmentation on test set — only apply the same preprocessing
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Detect structure: flat (yes/no) or split (Training/Testing)
subdirs = os.listdir(DATASET_PATH)
if 'Training' in subdirs or 'training' in subdirs:
    train_dir = os.path.join(DATASET_PATH, 'Training')
    test_dir  = os.path.join(DATASET_PATH, 'Testing')
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', seed=SEED, subset='training')
    val_gen   = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', seed=SEED, subset='validation')
    test_gen  = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False)
else:
    train_gen = train_datagen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', seed=SEED, subset='training')
    val_gen   = train_datagen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', seed=SEED, subset='validation')
    test_gen  = test_datagen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False)

CLASS_NAMES = list(train_gen.class_indices.keys())
NUM_CLASSES = len(CLASS_NAMES)
print(f"\nClasses: {CLASS_NAMES}")
print(f"Train samples : {train_gen.samples}")
print(f"Val samples   : {val_gen.samples}")
print(f"Test samples  : {test_gen.samples}")

# ============================================================
# CELL 6: Sample Images Visualization
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Sample Brain MRI Images', fontsize=16)
imgs, labels = next(train_gen)
for i, ax in enumerate(axes.flatten()):
    ax.imshow(imgs[i])
    ax.set_title(CLASS_NAMES[np.argmax(labels[i])])
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/sample_images.png', dpi=150)
plt.show()

# ============================================================
# CELL 7: Helper — Extract all images from a generator
# ============================================================
def extract_features_from_generator(generator, feature_model):
    """Extract ResNet50 features from an entire generator."""
    generator.reset()
    features, labels = [], []
    steps = len(generator)
    for i in range(steps):
        X_batch, y_batch = next(generator)
        feat = feature_model.predict(X_batch, verbose=0)
        features.append(feat)
        labels.append(y_batch)
        print(f"  Extracting batch {i+1}/{steps}", end='\r')
    print()
    return np.vstack(features), np.vstack(labels)

# ============================================================
# CELL 8: Build ResNet50 Feature Extractor (Approach 1)
# ============================================================
print("Building ResNet50 feature extractor...")
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze all layers

feature_extractor = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D()   # Outputs 2048-dim vector
], name='ResNet50_FeatureExtractor')

feature_extractor.summary()

# ============================================================
# CELL 9: Extract Features for SVM
# ============================================================
print("\nExtracting training features...")
t0 = time.time()
X_train_feat, y_train_raw = extract_features_from_generator(train_gen, feature_extractor)
X_val_feat,   y_val_raw   = extract_features_from_generator(val_gen,   feature_extractor)
X_test_feat,  y_test_raw  = extract_features_from_generator(test_gen,  feature_extractor)
feat_time = time.time() - t0

y_train_int = np.argmax(y_train_raw, axis=1)
y_val_int   = np.argmax(y_val_raw,   axis=1)
y_test_int  = np.argmax(y_test_raw,  axis=1)

print(f"\nFeature extraction done in {feat_time:.1f}s")
print(f"Feature shape per image: {X_train_feat.shape[1]}")

# Normalize features (important for SVM)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_feat)
X_val_sc   = scaler.transform(X_val_feat)
X_test_sc  = scaler.transform(X_test_feat)

# ============================================================
# CELL 10: APPROACH 1 — SVM (Linear Kernel)
# ============================================================
print("\n=== APPROACH 1: ResNet50 Features + SVM (Linear) ===")
t0 = time.time()
svm_model = SVC(kernel='linear', C=1.0, random_state=SEED)
svm_model.fit(X_train_sc, y_train_int)
svm_train_time = time.time() - t0
print(f"SVM training time: {svm_train_time:.2f}s")

y_pred_svm = svm_model.predict(X_test_sc)

svm_acc  = accuracy_score(y_test_int, y_pred_svm)
svm_prec = precision_score(y_test_int, y_pred_svm, average='weighted', zero_division=0)
svm_rec  = recall_score(y_test_int, y_pred_svm, average='weighted', zero_division=0)
svm_f1   = f1_score(y_test_int, y_pred_svm, average='weighted', zero_division=0)

print(f"\n--- Approach 1 Results (SVM) ---")
print(f"Accuracy  : {svm_acc:.4f}")
print(f"Precision : {svm_prec:.4f}")
print(f"Recall    : {svm_rec:.4f}")
print(f"F1-Score  : {svm_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_int, y_pred_svm, target_names=CLASS_NAMES))

# Confusion Matrix — Approach 1
cm_svm = confusion_matrix(y_test_int, y_pred_svm)
plt.figure(figsize=(6,5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix — Approach 1 (SVM)', fontsize=14)
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cm_svm.png', dpi=150)
plt.show()

# ============================================================
# CELL 11: APPROACH 2 — True End-to-End ResNet50 (2-Phase Fine-Tuning)
# ============================================================
# Phase 1: Train only the new top layers (base frozen)  → fast warm-up
# Phase 2: Unfreeze top 20 ResNet layers → true end-to-end fine-tuning
# This is the correct "End-to-End" approach required by the assignment.
# ============================================================
print("\n=== APPROACH 2: End-to-End ResNet50 (Fine-Tuning) ===")

train_gen.reset(); val_gen.reset(); test_gen.reset()

base_e2e = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(224, 224, 3))
base_e2e.trainable = False   # Start frozen for Phase 1

e2e_model = models.Sequential([
    base_e2e,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
], name='ResNet50_EndToEnd')

e2e_model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("--- Phase 1: Warm-up (top layers only, base frozen) ---")
early_stop = EarlyStopping(monitor='val_loss', patience=3,
                           restore_best_weights=True)
t0 = time.time()
history_p1 = e2e_model.fit(
    train_gen, validation_data=val_gen,
    epochs=5, callbacks=[early_stop], verbose=1
)

# ---- PHASE 2: Unfreeze top 50 ResNet50 layers ----
# ResNet50 is deep; 50 unfrozen layers allows the network to adapt its
# mid-level and high-level feature maps to MRI-specific patterns.
print("\n--- Phase 2: Fine-Tuning (top 50 ResNet layers unfrozen) ---")
for layer in base_e2e.layers[-50:]:
    layer.trainable = True

# Use a lower learning rate for fine-tuning to avoid destroying learned weights
e2e_model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop2 = EarlyStopping(monitor='val_loss', patience=4,
                            restore_best_weights=True)
train_gen.reset(); val_gen.reset()
history_p2 = e2e_model.fit(
    train_gen, validation_data=val_gen,
    epochs=EPOCHS, callbacks=[early_stop2], verbose=1
)
e2e_train_time = time.time() - t0
print(f"\nTotal End-to-End training time: {e2e_train_time:.1f}s")

# Merge both phase histories for learning curve plots
combined_history = {
    'accuracy':     history_p1.history['accuracy']     + history_p2.history['accuracy'],
    'val_accuracy': history_p1.history['val_accuracy'] + history_p2.history['val_accuracy'],
    'loss':         history_p1.history['loss']         + history_p2.history['loss'],
    'val_loss':     history_p1.history['val_loss']     + history_p2.history['val_loss'],
}

# ============================================================
# CELL 12: Learning Curves (Both Phases Combined)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
p1_epochs = len(history_p1.history['accuracy'])

ax1.plot(combined_history['accuracy'],     label='Train Accuracy', color='royalblue')
ax1.plot(combined_history['val_accuracy'], label='Val Accuracy',   color='orange')
ax1.axvline(x=p1_epochs - 0.5, color='gray', linestyle='--', label='Fine-Tuning Start')
ax1.set_title('Accuracy over Epochs (Phase 1 + Fine-Tuning)')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.legend(); ax1.grid(True)

ax2.plot(combined_history['loss'],     label='Train Loss', color='royalblue')
ax2.plot(combined_history['val_loss'], label='Val Loss',   color='orange')
ax2.axvline(x=p1_epochs - 0.5, color='gray', linestyle='--', label='Fine-Tuning Start')
ax2.set_title('Loss over Epochs (Phase 1 + Fine-Tuning)')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(True)

plt.suptitle('Approach 2: End-to-End ResNet50 Learning Curves', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/learning_curves.png', dpi=150)
plt.show()

# ============================================================
# CELL 13: Evaluate Approach 2
# ============================================================
test_gen.reset()
y_pred_e2e_prob = e2e_model.predict(test_gen, verbose=1)
y_pred_e2e = np.argmax(y_pred_e2e_prob, axis=1)

# Properly align true labels with predictions using filenames
# (avoids batch-padding mismatch at the last batch)
y_true_e2e  = test_gen.classes           # ground truth in generator order
y_pred_e2e  = y_pred_e2e[:len(y_true_e2e)]  # trim any padding from last batch
y_true_trimmed = y_true_e2e
y_pred_trimmed = y_pred_e2e

e2e_acc  = accuracy_score(y_true_trimmed, y_pred_trimmed)
e2e_prec = precision_score(y_true_trimmed, y_pred_trimmed, average='weighted', zero_division=0)
e2e_rec  = recall_score(y_true_trimmed, y_pred_trimmed, average='weighted', zero_division=0)
e2e_f1   = f1_score(y_true_trimmed, y_pred_trimmed, average='weighted', zero_division=0)

print(f"\n--- Approach 2 Results (End-to-End ResNet50) ---")
print(f"Accuracy  : {e2e_acc:.4f}")
print(f"Precision : {e2e_prec:.4f}")
print(f"Recall    : {e2e_rec:.4f}")
print(f"F1-Score  : {e2e_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_true_trimmed, y_pred_trimmed, target_names=CLASS_NAMES))

# Confusion Matrix — Approach 2
cm_e2e = confusion_matrix(y_true_trimmed, y_pred_trimmed)
plt.figure(figsize=(6,5))
sns.heatmap(cm_e2e, annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix — Approach 2 (End-to-End)', fontsize=14)
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cm_e2e.png', dpi=150)
plt.show()

# ============================================================
# CELL 14: COMPARATIVE ANALYSIS — Bar Chart
# ============================================================
metrics      = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
svm_scores   = [svm_acc, svm_prec, svm_rec, svm_f1]
e2e_scores   = [e2e_acc, e2e_prec, e2e_rec, e2e_f1]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, svm_scores, width, label='Approach 1 (SVM)',
               color='steelblue', alpha=0.85)
bars2 = ax.bar(x + width/2, e2e_scores, width, label='Approach 2 (End-to-End)',
               color='darkorange', alpha=0.85)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparative Analysis: SVM vs End-to-End ResNet50\n(Brain MRI Tumor Detection)',
             fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/comparison_bar_chart.png', dpi=150)
plt.show()

# ============================================================
# CELL 15: Training Time Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
times  = [svm_train_time, e2e_train_time]
labels = ['Approach 1\n(Feature Extraction + SVM)', 'Approach 2\n(End-to-End ResNet50)']
colors = ['steelblue', 'darkorange']
bars = ax.bar(labels, times, color=colors, alpha=0.85, width=0.4)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Training Time Comparison', fontsize=14)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}s', ha='center', va='bottom', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_time_comparison.png', dpi=150)
plt.show()

# ============================================================
# CELL 16: Hyperparameter Table (Required by Req 2.3)
# ============================================================
print("\n" + "="*70)
print("              HYPERPARAMETER TABLE (Requirement 2.3)")
print("="*70)
print(f"{'Hyperparameter':<30} {'Approach 1 (SVM)':<22} {'Approach 2 (E2E)'}")
print("-"*70)
hp_rows = [
    ('Base CNN',           'ResNet50 (frozen)',      'ResNet50 (fine-tuned)'),
    ('Input Image Size',   '224 x 224',              '224 x 224'),
    ('Color Channels',     'RGB (3)',                 'RGB (3)'),
    ('Batch Size',         'N/A',                    str(BATCH_SIZE)),
    ('Epochs',             'N/A',                    f'Phase1=5, Phase2={EPOCHS}'),
    ('Learning Rate',      'N/A',                    f'{LEARNING_RATE} / {LEARNING_RATE/10}'),
    ('Optimizer',          'N/A',                    'Adam'),
    ('Loss Function',      'N/A',                    'Categorical Crossentropy'),
    ('Classifier',         'SVM',                    'Softmax Dense Layer'),
    ('SVM Kernel',         'Linear',                 'N/A'),
    ('SVM C',              '1.0',                    'N/A'),
    ('Feature Dim',        '2048 (ResNet50 output)', 'N/A'),
    ('Feature Scaling',    'StandardScaler',         'N/A'),
    ('Dropout',            'N/A',                    '0.5'),
    ('Early Stopping',     'N/A',                    'patience=4'),
    ('Unfrozen Layers',    '0 (all frozen)',          'Top 50 ResNet layers'),
]
for row in hp_rows:
    print(f"{row[0]:<30} {row[1]:<22} {row[2]}")
print("="*70)

# ============================================================
# CELL 17: Final Results Summary
# ============================================================
print("\n" + "="*60)
print("       FINAL RESULTS SUMMARY")
print("="*60)
print(f"{'Metric':<20} {'Approach 1 (SVM)':<22} {'Approach 2 (E2E)'}")
print("-"*60)
for m, s, e in zip(metrics, svm_scores, e2e_scores):
    print(f"{m:<20} {s:<22.4f} {e:.4f}")
print("-"*60)
print(f"{'Train Time (s)':<20} {svm_train_time:<22.2f} {e2e_train_time:.2f}")
print("="*60)

print(f"\n✅ All charts saved to Google Drive: {OUTPUT_DIR}")
print("  - sample_images.png")
print("  - cm_svm.png")
print("  - cm_e2e.png")
print("  - learning_curves.png")
print("  - comparison_bar_chart.png")
print("  - training_time_comparison.png")
print("\nOpen your Google Drive → deep-learning → project_outputs to find them!")
