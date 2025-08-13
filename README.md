# CNN vs CNN+SVM for MNIST Classification

This project compares two approaches for handwritten digit recognition using the **MNIST** dataset:

1. **CNN** – A standard Convolutional Neural Network with a softmax classifier.
2. **CNN + SVM** – A hybrid model where the CNN acts as a fixed feature extractor, and a **Support Vector Machine** (SVM) performs the final classification.

The goal is to see if replacing the softmax layer with an SVM improves accuracy.

---

## Features
- Train a CNN from scratch using PyTorch
- Use the trained CNN to extract 128-dimensional feature vectors
- Train an SVM classifier on CNN features using scikit-learn
- Compare accuracy and confusion matrices for both approaches
- Visualize learned features using **t-SNE**

---

## Project Structure
```

cnn_svm.ipynb        # Main experiment notebook
models/              # Saved CNN and SVM models
data/                # MNIST dataset (downloaded automatically)

````

---

## 🛠 Requirements
Install the required Python packages:
```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas joblib
````

---

## How to Run

Run the experiment:

```bash
# use interactive notebook like google collab
```

This will:

1. Download MNIST and prepare data loaders.
2. Train a CNN with a softmax classifier.
3. Save the CNN feature extractor.
4. Train an SVM on extracted CNN features.
5. Display confusion matrices for both models.
6. Run t-SNE visualization of CNN features.

---

## Model Details

### CNN Architecture

* **Conv1:** 1 input channel → 32 output channels, kernel 3×3, padding 1, ReLU, MaxPool 2×2
* **Conv2:** 32 input channels → 64 output channels, kernel 3×3, padding 1, ReLU, MaxPool 2×2
* **Flatten + FC:** Fully connected to 128 features
* **Classifier (Pure CNN):** Fully connected to 10 outputs (digits 0–9)

### SVM Classifier

* **Kernel:** Linear
* **C:** 1.0
* **Features:** 128-dimensional vectors from CNN

---

## Results

| Model              | Test Accuracy |
| ------------------ | ------------- |
| Pure CNN (Softmax) | \~99.06%      |
| CNN + SVM          | \~99.15%      |

Both models perform very well. The hybrid CNN+SVM approach offers a slight improvement.

---

## Visualizations

* **Confusion Matrices** for both models
* **t-SNE Plot** showing how CNN features cluster by digit class


## Conclusion

* A plain CNN already achieves excellent accuracy on MNIST.
* Adding an SVM classifier to CNN features can give a small performance boost when the features are well-separated.
