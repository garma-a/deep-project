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

1. **Resizing:** All MRI images were resized to **224 × 224 pixels** to match the required input dimensions of the ResNet50 architecture, which was originally trained on ImageNet images of this size.

2. **Pixel Normalization:** All pixel values were normalized from the range [0, 255] to the range **[0, 1]** by dividing by 255:
   ```
   pixel_normalized = pixel_value / 255.0
   ```
   This normalization ensures consistent gradient magnitudes during training, accelerates convergence, and prevents neurons from saturating due to large input values.

3. **RGB Conversion:** MRI images (originally grayscale) were loaded as RGB (3-channel) images to be compatible with the ResNet50 architecture, which expects 3-channel input.

---

### 1.4 Data Augmentation

The following augmentation techniques were applied **only to the training set** to artificially expand the dataset and improve model generalization:

| Technique | Parameter | Justification |
|---|---|---|
| **Rotation** | ±15° | MRI scans are taken with patients in slightly different head positions. Rotation simulates this variability and prevents the model from overfitting to a fixed orientation. |
| **Horizontal Flip** | Enabled | Brain tumors can appear on either hemisphere (left or right side). Flipping doubles the effective training data and forces the model to be hemisphere-agnostic. |
| **Zoom** | ±10% | Simulates varying distances between the MRI scanner and the patient's head, producing images of slightly different apparent scales. |
| **Width Shift** | ±10% | Accounts for slight lateral positioning differences between scans from different hospitals or scanners. |
| **Height Shift** | ±10% | Accounts for slight vertical positioning differences in scan acquisition. |

**Overall Justification:**  
Medical imaging datasets are inherently smaller than general computer vision datasets (like ImageNet) due to the cost and privacy constraints of acquiring medical data. Augmentation is essential to prevent overfitting, especially when using a deep model like ResNet50. The selected techniques are conservative (±15° rotation, ±10% zoom/shifts) to preserve the medical validity of the images — extreme augmentations (e.g., 180° rotation or heavy distortion) could create unrealistic MRI appearances that do not reflect real clinical data.

> **Note:** Augmentation was NOT applied to the validation or test sets, as these must reflect real-world data conditions to provide an unbiased performance evaluation.

---

### 1.5 Data Splitting

The dataset was pre-split by the original authors into Training and Testing folders. An additional validation split was extracted from the training set:

| Split | Images | Percentage |
|---|---|---|
| **Training** | 4,480 | ~62% |
| **Validation** | 1,120 | ~16% |
| **Testing** | 1,600 | ~22% |
| **Total** | **7,200** | 100% |

**Method:** An 80/20 split was applied to the Training folder (80% training, 20% validation) using Keras `ImageDataGenerator`'s `validation_split=0.2` parameter. The Testing folder was used as the held-out test set, ensuring the model never sees test images during training or hyperparameter tuning.

---

## Requirement 2: DL Model Selection

### 2.1 Selected Architecture: ResNet50

**ResNet50** (Residual Network with 50 layers) was selected as the backbone CNN architecture for both approaches, pre-trained on the ImageNet dataset (1.2 million images, 1000 classes).

---

### 2.2 Technical Justification for ResNet50

#### The Core Problem: Vanishing Gradients in Deep Networks
As neural networks become deeper, training them becomes increasingly difficult. During backpropagation, gradients are multiplied through many layers. When these gradients are small (< 1), they become exponentially smaller as they propagate backward through 50+ layers — effectively "vanishing" and causing early layers to learn extremely slowly or not at all.

#### ResNet50's Solution: Residual (Skip) Connections
ResNet50 introduces **residual connections** (also called skip connections) that bypass one or more layers:

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

Instead of learning `F(x)` from scratch, each residual block learns `F(x) + x`, where `x` is the original input passed directly. This means:
- If a layer is not useful, it can simply learn `F(x) = 0`, making the block an identity function
- Gradients can flow directly through the skip connection, solving the vanishing gradient problem
- Networks can be made much deeper (50, 101, 152 layers) without degradation

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
| Total Parameters | ~25.6 million |

**Reference:**  
> He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778. https://doi.org/10.1109/CVPR.2016.90

#### Why ResNet50 for Brain MRI?
1. **Transfer Learning:** Pre-trained on 1.2M ImageNet images, ResNet50 has already learned to detect edges, textures, shapes, and complex patterns — all of which are relevant for distinguishing tumor morphologies in MRI scans.
2. **Depth vs. Efficiency:** ResNet50 is deep enough to capture complex hierarchical features (tumor shapes, boundaries, contrast) while being computationally manageable compared to deeper variants (ResNet101, ResNet152).
3. **Proven Medical Imaging Performance:** ResNet50 is extensively cited in the medical imaging literature for tasks ranging from tumor detection to diabetic retinopathy classification.
4. **2048-Dimensional Features:** The final feature vector from `GlobalAveragePooling2D` is 2048 dimensions — rich enough to provide a powerful representation for the SVM classifier in Approach 1.

---

### 2.3 Hyperparameter Table

| Hyperparameter | Approach 1 (ResNet50 + SVM) | Approach 2 (End-to-End ResNet50) |
|---|---|---|
| **Base CNN** | ResNet50 (all layers frozen) | ResNet50 (fine-tuned) |
| **Pre-trained Weights** | ImageNet | ImageNet |
| **Input Image Size** | 224 × 224 | 224 × 224 |
| **Color Channels** | RGB (3) | RGB (3) |
| **Batch Size** | N/A (features extracted) | 32 |
| **Phase 1 Epochs** | N/A | 5 (warm-up) |
| **Phase 2 Epochs** | N/A | Up to 15 (fine-tuning) |
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
| **Unfrozen Layers** | 0 (fully frozen) | Top 20 ResNet50 layers |
| **Random Seed** | 42 | 42 |

---

## Requirement 3: Implementation Framework

### Approach 1: DL-Based Feature Learning + SVM Classifier

#### Step 1: Feature Extraction
The ResNet50 model (pre-trained on ImageNet) is loaded with `include_top=False` and `weights='imagenet'`. All layers are frozen (`trainable=False`). A `GlobalAveragePooling2D` layer is appended to convert the final convolutional feature maps (7×7×2048) into a flat 2048-dimensional vector for each image.

**Why freeze the layers?**  
In Approach 1, ResNet50 acts purely as a "feature extractor." We are not asking it to learn anything new — we are using its existing ImageNet knowledge to describe our MRI images as numerical vectors. Freezing ensures reproducibility and speed.

#### Step 2: Feature Normalization
The extracted 2048-dimensional feature vectors are normalized using `StandardScaler` (zero mean, unit variance). This is critical for SVM performance — SVMs are distance-based models and are highly sensitive to feature scale.

#### Step 3: SVM Classification
A Support Vector Machine with a **linear kernel** is trained on the normalized feature vectors.

**Why SVM with Linear Kernel?**
- The feature space is 2048-dimensional. SVMs are mathematically designed to find optimal decision boundaries in high-dimensional spaces through margin maximization.
- A linear kernel is sufficient because ResNet50 features are already highly non-linear transformations of the raw pixels. The features are linearly separable in the 2048-dimensional space.
- Linear SVM is computationally faster than RBF kernels in high-dimensional settings.

#### Performance Results (Approach 1):
*(To be filled after training completes)*

| Metric | Score |
|---|---|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-Score | — |

**Confusion Matrix:** *(insert cm_svm.png)*

---

### Approach 2: End-to-End Deep Learning (Fine-Tuned ResNet50)

#### Two-Phase Training Strategy

**Phase 1 — Warm-up (5 epochs, base frozen):**  
The ResNet50 base remains frozen while only the newly added classification head (Dense → Dropout → Softmax) is trained. This prevents the randomly initialized head from destroying the pre-trained ImageNet weights through large gradient updates in early training.

**Phase 2 — Fine-Tuning (up to 15 epochs, top 20 layers unfrozen):**  
The top 20 layers of ResNet50 are unfrozen and allowed to adapt their weights to the MRI domain. A reduced learning rate (1e-5 vs. 1e-4 in Phase 1) is used to make small, careful adjustments that preserve the general feature knowledge while adapting to brain tumor-specific features.

**Classification Head:**
```
ResNet50 Base (224×224×3 input)
    ↓
GlobalAveragePooling2D → 2048-dim vector
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.5)  ← prevents overfitting
    ↓
Dense(4, activation='softmax')  ← 4 tumor classes
```

**Why End-to-End?**  
End-to-end training allows the model to optimize all components simultaneously for the specific task. The fine-tuned layers adapt ResNet50's ImageNet features (which encode natural image statistics) toward MRI-specific features (tumor morphology, contrast, texture in grayscale medical images).

#### Performance Results (Approach 2):
*(To be filled after training completes)*

| Metric | Score |
|---|---|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-Score | — |

**Confusion Matrix:** *(insert cm_e2e.png)*  
**Learning Curves:** *(insert learning_curves.png)*

---

## Requirement 4: Comparative Analysis and Insights

### 4.1 Comparative Analysis

*(Insert comparison_bar_chart.png and training_time_comparison.png here)*

| Metric | Approach 1 (SVM) | Approach 2 (End-to-End) | Winner |
|---|---|---|---|
| Accuracy | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |
| F1-Score | — | — | — |
| Training Time | — (seconds) | — (seconds) | — |

---

### 4.2 Conclusion

*(To be filled after training completes with actual numbers)*

**i. Performance Comparison:**  
[Will be filled with actual results]

**ii. Advantages and Limitations:**

| | Approach 1 (DL Features + SVM) | Approach 2 (End-to-End) |
|---|---|---|
| **Advantages** | Fast training, no GPU needed for classifier, interpretable, robust in low-data scenarios | Higher accuracy, adapts to target domain, single unified pipeline |
| **Limitations** | Features not optimized for MRI domain (ImageNet bias), no joint optimization | Slower training, requires GPU, risk of overfitting on small datasets |

**iii. Training Time Efficiency:**  
Approach 1 is significantly faster — feature extraction is a single forward pass, and SVM training takes seconds. Approach 2 requires full backpropagation through multiple epochs over the entire training set.

**iv. Recommendations:**

| Environment | Recommended Approach | Reason |
|---|---|---|
| **Resource-constrained** (mobile, edge) | **Approach 1 (SVM)** | Features extracted once offline; lightweight SVM inference requires no GPU; suitable for deployment on embedded systems |
| **High-performance** (cloud, research) | **Approach 2 (End-to-End)** | Full GPU utilization, maximum accuracy, domain-specific adaptation justifies the computational cost |

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR 2016. https://doi.org/10.1109/CVPR.2016.90

2. Bhuvaji, S. (2020). *Brain Tumor Classification (MRI).* Kaggle Dataset. https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

3. Cortes, C., & Vapnik, V. (1995). *Support-vector networks.* Machine Learning, 20(3), 273–297.

4. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.

5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press. https://www.deeplearningbook.org
