# Deep Learning Project Report
## CAI3105/CS460 — 12th Week Project
### End-to-End DL Classification vs. DL-Based Feature Learning

---

## Requirement 1: Dataset Selection and Technical Specifications

### 1.1 Dataset Metadata

| Field | Details | 
|---|---|
| **Dataset Name** | Brain Tumor Classification (MRI) |
| **Source** | Kaggle — Sartaj Bhuvaji |
| **Kaggle Link** | https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri |
| **Problem Domain** | Medical Imaging — Brain Tumor Detection & Classification |
| **Total Samples (N)** | **7,200 MRI images** |

**Justification for Dataset Selection:**  
The Brain Tumor MRI dataset was selected because it represents a high-impact, real-world medical diagnosis problem. Early and accurate detection of brain tumors is critical for patient outcomes. The dataset provides a challenging multi-class classification task across four distinct tumor types, making it ideal for evaluating the comparative performance of DL-based feature extraction versus end-to-end deep learning.

---

### 1.2 Technical Specifications

| Specification | Value |
|---|---|
| **Original Image Resolution** | Variable (different MRI scan resolutions) |
| **Resized Resolution** | **224 × 224 pixels** |
| **Color Channels** | **RGB (3 channels)** — converted from grayscale MRI scans to match ResNet50's expected input format |
| **Total Number of Classes** | **4 classes** |

**Class Labels:**

| Class | Description |
|---|---|
| `glioma` | Glioma tumor — most common primary brain tumor |
| `meningioma` | Meningioma tumor — arises from the meninges |
| `notumor` | Healthy brain — no tumor present |
| `pituitary` | Pituitary tumor — grows in the pituitary gland |

---

### 1.3 Data Preprocessing

The following preprocessing steps were applied to all images before being fed into the model:

1. **Resizing:** All MRI images were resized to **224 × 224 pixels** to match the required input dimensions of the ResNet50 architecture.

2. **ResNet50-Specific Normalization:** Instead of a simple `/255` rescaling, the official `preprocess_input` function from `tensorflow.keras.applications.resnet50` was applied. This function performs:
   - Conversion from RGB to BGR channel ordering
   - Channel-wise mean subtraction using ImageNet statistics: `[103.939, 116.779, 123.68]`
   
   This is critical because pretrained CNNs are extremely sensitive to preprocessing mismatch — the model's weights were calibrated on this exact input distribution during ImageNet training.

   ```python
   from tensorflow.keras.applications.resnet50 import preprocess_input
   # Applied via ImageDataGenerator(preprocessing_function=preprocess_input)
   ```

3. **RGB Conversion:** MRI images (originally grayscale) were loaded as RGB (3-channel) images to be compatible with ResNet50's input expectations.

---

### 1.4 Data Augmentation

The following augmentation techniques were applied **only to the training set**:

| Technique | Parameter | Justification |
|---|---|---|
| **Rotation** | ±15° | MRI scans are taken with patients in slightly different head positions. Rotation simulates this variability and prevents the model from overfitting to a fixed orientation. |
| **Horizontal Flip** | **Disabled** | Left/right brain hemisphere orientation carries clinical meaning in MRI scans. Flipping can create anatomically unrealistic samples and mislead the model about tumor laterality. |
| **Zoom** | ±10% | Simulates varying distances between the MRI scanner and the patient's head. |
| **Width Shift** | ±10% | Accounts for slight lateral positioning differences between scans. |
| **Height Shift** | ±10% | Accounts for slight vertical positioning differences in scan acquisition. |

**Overall Justification:**  
Medical imaging datasets are inherently smaller than general computer vision datasets due to the cost and privacy constraints of acquiring medical data. Augmentation is essential to prevent overfitting. The techniques are deliberately conservative — extreme augmentations could create clinically unrealistic images. Horizontal flip was explicitly **disabled** because left-right orientation is diagnostically meaningful in brain MRI, unlike natural image datasets (e.g., cats/dogs) where flipping is harmless.

> **Note:** Augmentation was NOT applied to the validation or test sets, ensuring unbiased performance evaluation.

---

### 1.5 Data Splitting

The dataset was pre-split by the original authors into Training and Testing folders. An additional validation split was extracted from the training set:

| Split | Images | Percentage |
|---|---|---|
| **Training** | 4,480 | ~62% |
| **Validation** | 1,120 | ~16% |
| **Testing** | 1,600 | ~22% |
| **Total** | **7,200** | 100% |

**Method:** An 80/20 split was applied to the Training folder using Keras `ImageDataGenerator`'s `validation_split=0.2`. The Testing folder was used as the held-out test set, ensuring the model never sees test images during training or hyperparameter tuning.

---

## Requirement 2: DL Model Selection

### 2.1 Selected Architecture: ResNet50

**ResNet50** (Residual Network with 50 layers) was selected as the backbone CNN architecture for both approaches, pre-trained on the ImageNet dataset (1.2 million images, 1000 classes).

---

### 2.2 Technical Justification for ResNet50

#### The Core Problem: Vanishing Gradients in Deep Networks
As neural networks become deeper, gradients shrink exponentially during backpropagation through many layers — effectively "vanishing" and causing early layers to learn extremely slowly or not at all.

#### ResNet50's Solution: Residual (Skip) Connections
ResNet50 introduces **residual connections** (skip connections) that bypass one or more layers:

```
Input (x)
    │
    ├──────────────────────┐  ← Skip Connection
    │                      │
   [Conv → BN → ReLU]      │
   [Conv → BN → ReLU]      │
   [Conv → BN]             │
    │                      │
    └──────── + ───────────┘  ← Element-wise addition
              │
            ReLU
              │
           Output (F(x) + x)
```

Each residual block learns `F(x) + x` — if a layer isn't useful, it can learn `F(x) = 0` (identity). Gradients also flow directly through skip connections, solving the vanishing gradient problem and enabling training of 50+ layer networks without degradation.

#### Architecture Summary

| Component | Details |
|---|---|
| Input | 224 × 224 × 3 |
| Initial Conv | 7×7, 64 filters, stride 2 |
| Max Pooling | 3×3, stride 2 |
| Residual Blocks | 4 stages (conv2_x to conv5_x) |
| Stage 1 | 3 blocks, 64→256 channels |
| Stage 2 | 4 blocks, 128→512 channels |
| Stage 3 | 6 blocks, 256→1024 channels |
| Stage 4 | 3 blocks, 512→2048 channels |
| Global Avg Pool | 2048-dimensional feature vector |
| Total Parameters | ~23.6 million |

**Reference:**  
> He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778. https://doi.org/10.1109/CVPR.2016.90

#### Why ResNet50 for Brain MRI?
1. **Transfer Learning:** Pre-trained on 1.2M ImageNet images, ResNet50 has already learned edges, textures, shapes, and complex patterns — all relevant for distinguishing tumor morphologies.
2. **Depth vs. Efficiency:** Deep enough to capture hierarchical features (tumor shapes, boundaries, contrast) while remaining computationally manageable.
3. **Proven Medical Imaging Performance:** Extensively cited in the medical imaging literature for tumor detection and classification tasks.
4. **2048-Dimensional Features:** Rich enough representation for the SVM classifier in Approach 1.

---

### 2.3 Hyperparameter Table

| Hyperparameter | Approach 1 (ResNet50 + SVM) | Approach 2 (End-to-End ResNet50) |
|---|---|---|
| **Base CNN** | ResNet50 (all layers frozen) | ResNet50 (fine-tuned) |
| **Pre-trained Weights** | ImageNet | ImageNet |
| **Preprocessing** | `preprocess_input` (ResNet50) | `preprocess_input` (ResNet50) |
| **Input Image Size** | 224 × 224 | 224 × 224 |
| **Color Channels** | RGB (3) | RGB (3) |
| **Batch Size** | N/A (features extracted) | 32 |
| **Phase 1 Epochs** | N/A | 5 (warm-up) |
| **Phase 2 Epochs** | N/A | Up to 25 (fine-tuning) |
| **Phase 1 Learning Rate** | N/A | 0.0001 |
| **Phase 2 Learning Rate** | N/A | 0.00001 (1/10 of Phase 1) |
| **Optimizer** | N/A | Adam |
| **Loss Function** | N/A | Categorical Crossentropy |
| **Top Classifier** | Linear SVM | Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax) |
| **SVM Kernel** | **Linear** | N/A |
| **SVM Regularization (C)** | 1.0 | N/A |
| **Feature Dimensionality** | 2048 (ResNet50 GAP output) | N/A |
| **Feature Scaling** | StandardScaler (zero mean, unit variance) | N/A |
| **Dropout Rate** | N/A | 0.5 |
| **Early Stopping** | N/A | patience=3 (Phase 1), patience=4 (Phase 2) |
| **Unfrozen Layers** | 0 (fully frozen) | Top 50 ResNet50 layers |
| **Horizontal Flip** | No | No |
| **Random Seed** | 42 | 42 |

---

## Requirement 3: Implementation Framework

### Approach 1: DL-Based Feature Learning + SVM Classifier

#### Step 1: Feature Extraction
The ResNet50 model (pre-trained on ImageNet) is loaded with `include_top=False` and `weights='imagenet'`. All layers are frozen (`trainable=False`). A `GlobalAveragePooling2D` layer converts the final feature maps (7×7×2048) into a flat **2048-dimensional vector** per image.

Feature extraction completed in **118.7 seconds** across all 3 splits (train/val/test).

#### Step 2: Feature Normalization
The 2048-dimensional vectors are normalized using `StandardScaler` (zero mean, unit variance). This is critical — SVMs are distance-based and sensitive to feature scale.

#### Step 3: SVM Classification
A Support Vector Machine with a **linear kernel** (C=1.0) is trained on the normalized features. SVM training completed in **5.95 seconds**.

#### Performance Results (Approach 1):

| Metric | Score |
|---|---|
| **Accuracy** | **87.44%** |
| **Precision** | **87.23%** |
| **Recall** | **87.44%** |
| **F1-Score** | **87.14%** |
| **Training Time** | **5.95 seconds** |

**Per-class Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| glioma | 0.87 | 0.74 | 0.80 | 400 |
| meningioma | 0.80 | 0.78 | 0.79 | 400 |
| notumor | 0.91 | 0.99 | 0.95 | 400 |
| pituitary | 0.91 | 0.97 | 0.94 | 400 |
| **Weighted Avg** | **0.87** | **0.87** | **0.87** | 1600 |

---

### Approach 2: End-to-End Deep Learning (Fine-Tuned ResNet50)

#### Two-Phase Training Strategy

**Phase 1 — Warm-up (5 epochs, base frozen):**  
Only the classification head (Dense → Dropout → Softmax) is trained. This prevents randomly initialized weights from destroying pre-trained ImageNet features.

| Epoch | Train Acc | Val Acc | Val Loss |
|---|---|---|---|
| 1 | 64.46% | 83.48% | 0.4873 |
| 2 | 81.41% | 87.14% | 0.3669 |
| 3 | 84.60% | 87.05% | 0.3421 |
| 4 | 86.45% | 89.38% | 0.3019 |
| 5 | 88.06% | **90.80%** | **0.2893** |

**Phase 2 — Fine-Tuning (top 50 layers unfrozen, patience=4):**  
The top 50 layers of ResNet50 are unfrozen with learning rate reduced to 1e-5. EarlyStopping triggered at epoch 11 (val_loss stopped improving after patience=4).

| Epoch | Train Acc | Val Acc | Val Loss |
|---|---|---|---|
| 1 | 85.89% | 91.96% | 0.2240 |
| 2 | 92.10% | 93.39% | 0.1836 |
| 3 | 94.04% | 94.29% | 0.1656 |
| 4 | 95.45% | 95.45% | 0.1348 |
| 5 | 96.12% | 95.27% | 0.1239 |
| 6 | 96.96% | 96.16% | 0.1083 |
| 7 | 97.66% | **96.96%** | **0.0966** ← best |
| 8 | 98.39% | 96.96% | 0.1046 |
| 9 | 98.44% | 96.79% | 0.1015 |
| 10 | 98.42% | 96.52% | 0.1130 |
| 11 | 98.84% | 96.70% | 0.0997 ← stopped |

**Total E2E training time: 1,309.4 seconds (~22 minutes)**

**Classification Head:**
```
ResNet50 Base (224×224×3 input)
    ↓
GlobalAveragePooling2D → 2048-dim vector
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(4, activation='softmax')  ← 4 tumor classes
```

#### Performance Results (Approach 2):

| Metric | Score |
|---|---|
| **Accuracy** | **92.69%** |
| **Precision** | **92.88%** |
| **Recall** | **92.69%** |
| **F1-Score** | **92.54%** |
| **Training Time** | **1,309.42 seconds** |

**Per-class Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| glioma | 0.96 | 0.80 | 0.87 | 400 |
| meningioma | 0.87 | 0.93 | 0.90 | 400 |
| notumor | 0.93 | 0.99 | 0.96 | 400 |
| pituitary | 0.95 | 0.99 | 0.97 | 400 |
| **Weighted Avg** | **0.93** | **0.93** | **0.93** | 1600 |

---

## Requirement 4: Comparative Analysis and Insights

### 4.1 Comparative Analysis

| Metric | Approach 1 (SVM) | Approach 2 (End-to-End) | Winner |
|---|---|---|---|
| **Accuracy** | 87.44% | **92.69%** | ✅ E2E (+5.25%) |
| **Precision** | 87.23% | **92.88%** | ✅ E2E (+5.65%) |
| **Recall** | 87.44% | **92.69%** | ✅ E2E (+5.25%) |
| **F1-Score** | 87.14% | **92.54%** | ✅ E2E (+5.40%) |
| **Training Time** | **5.95s** | 1,309.42s | ✅ SVM (220× faster) |

---

### 4.2 Conclusion

**i. Performance Comparison:**  
The End-to-End fine-tuned ResNet50 (92.69%) significantly outperformed the SVM hybrid pipeline (87.44%) by **5.25 percentage points** in accuracy. This gap is consistent across all metrics (precision, recall, F1). Fine-tuning the top 50 ResNet50 layers allowed the network to adapt its mid-level and high-level feature detectors to MRI-specific characteristics — tumor morphology, contrast boundaries, and tissue textures — which a frozen extractor cannot capture, as it is constrained to ImageNet-learned representations.

**ii. Advantages and Limitations:**

| | Approach 1 (DL Features + SVM) | Approach 2 (End-to-End) |
|---|---|---|
| **Advantages** | 220× faster training (5.95s), no GPU required for the classifier, simple and interpretable pipeline, robust in low-data scenarios | Higher accuracy (+5.25%), task-specific feature adaptation, single unified optimized pipeline |
| **Limitations** | Features are fixed to ImageNet domain — not optimized for MRI; no joint optimization between extractor and classifier | 22-minute training time, requires GPU, risk of overfitting if dataset is very small |

**iii. Training Time Efficiency:**  
Approach 1 was dramatically faster — feature extraction is a single forward pass (~119s for all images), and SVM training took only 5.95 seconds. Approach 2 required 1,309 seconds (~22 minutes) of full backpropagation over 16 epochs (5 warm-up + 11 fine-tuning before EarlyStopping). Approach 1 is approximately **220× faster** overall.

**iv. Recommendations:**

| Environment | Recommended Approach | Reason |
|---|---|---|
| **Resource-constrained** (mobile, edge, embedded) | **Approach 1 (SVM)** | 87.44% accuracy with near-zero inference cost; features can be extracted once offline; SVM runs without GPU; suitable for deployment on low-power devices |
| **High-performance** (cloud, research, hospital server) | **Approach 2 (End-to-End)** | 92.69% accuracy is critical in medical diagnostics where misclassification has serious consequences; GPU cost is justified by the 5.25% accuracy gain |

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR 2016. https://doi.org/10.1109/CVPR.2016.90

2. Bhuvaji, S. (2020). *Brain Tumor Classification (MRI).* Kaggle Dataset. https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

3. Cortes, C., & Vapnik, V. (1995). *Support-vector networks.* Machine Learning, 20(3), 273–297.

4. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.

5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press. https://www.deeplearningbook.org

6. TensorFlow/Keras Documentation. *ResNet50 preprocess_input.* https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input
