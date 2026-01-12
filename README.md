# ML-Project-2026
# Sentiment Classification on Noisy Customer Reviews: Classical vs Neural Network Approaches

**Author:** Elif Şevval Akdeniz  
**Institution:** Universiteit Antwerpen, Faculty of Arts  
**Course:** Digital Text Analysis - Machine Learning  
**Date:** January 2026


## 📋 Project Overview

This project compares classical machine learning and neural network approaches for sentiment classification on noisy, real-world customer reviews. Using 33,396 McDonald's reviews from Google Reviews (55.6% containing textual noise), we evaluate four models: **Logistic Regression**, **Support Vector Machine (SVM)**, **Random Forest**, and **RNN-LSTM** with class-weighted training.

### Key Research Questions
1. How do classical ML algorithms compare to neural networks in overall performance on noisy customer reviews?
2. How does noise severity differentially affect these algorithmic families?

---

## 🎯 Key Findings

- **Classical models substantially outperform neural networks** on this dataset
- **SVM achieves the best performance**: 81.1% accuracy, macro F1-score 0.73
- **RNN-LSTM underperforms despite optimization**: 71.0% accuracy, macro F1-score 0.66 (7-point gap)
- **Counter-intuitive result**: Aggressive preprocessing degrades neural network performance more than classical models
- **Practical implication**: For datasets with limited samples (20k-50k) and class imbalance, classical ML offers superior performance

### Performance Summary

| Model | Macro F1 | Accuracy | Weighted F1 | Neutral F1 |
|-------|----------|----------|-------------|------------|
| **SVM** | **0.73** | **81.1%** | **0.801** | **0.47** |
| Logistic Regression | 0.72 | 81.3% | 0.799 | 0.45 |
| Random Forest | 0.67 | 78.0% | 0.755 | 0.38 |
| RNN-LSTM | 0.66 | 71.0% | 0.733 | 0.39 |

---

## Dataset

- **Source:** McDonald's customer reviews from Google Reviews (United States)
- **Size:** 33,396 reviews
- **Sentiment Distribution:**
  - Positive (4-5 stars): 48.1%
  - Negative (1-2 stars): 37.5%
  - Neutral (3 stars): 14.4% *(minority class)*
- **Noise Characteristics:**
  - 55.6% of reviews contain noise
  - Noise types: abbreviations (43.4%), repeated characters (8.4%), slang (9.1%), ALL CAPS (6.6%), typos (5.5%)
  - Noise levels: Clean (44.4%), Low (25.0%), Medium (15.7%), High (14.9%)
- **Text Properties:**
  - Mean length: 22.1 words
  - Median length: 11 words

---

## Methodology

### Preprocessing
**Minimal preprocessing** was deliberately chosen to preserve authentic noise patterns for robustness testing:
- Lowercase conversion
- URL/email removal
- **Preserved:** punctuation, emoticons, slang, typos, abbreviations, repeated characters, stop words

Supplementary analysis (Appendix B) validates this choice by showing aggressive preprocessing degrades neural network performance substantially.

### Models

#### Classical ML (TF-IDF + Algorithms)
- **Logistic Regression:** L2 regularization (C=1.0), L-BFGS optimization
- **Support Vector Machine:** Linear kernel, margin maximization (C=1.0)
- **Random Forest:** 100 trees, max depth 30, bootstrap sampling

#### Neural Network
- **RNN-LSTM Architecture:**
  - Embedding layer: 128 dimensions
  - LSTM layer: 64 hidden units
  - Dense output: 3 units (softmax)
  - Dropout: 0.3
- **Class-weighted loss:** Neutral class receives 3.33× higher penalty
- **Training:** Early stopping (patience 5), learning rate scheduling, gradient clipping
- **Optimization:** Adam optimizer (lr=0.001), batch size 32

### Evaluation Metrics
- **Primary:** Macro-averaged F1-score (accounts for class imbalance)
- **Secondary:** Weighted F1-score, accuracy, precision, recall
- **Data split:** 60% train, 20% validation, 20% test (stratified)

---

## Results

### Overall Performance
Classical models consistently outperform the neural network:
- **Performance gap:** 7 points in macro F1-score (SVM 0.73 vs RNN 0.66)
- **Accuracy gap:** 10.1 percentage points (SVM 81.1% vs RNN 71.0%)

### Per-Class Analysis
| Model | Negative F1 | Neutral F1 | Positive F1 | Macro F1 |
|-------|-------------|------------|-------------|----------|
| SVM | 0.85 | **0.47** | 0.87 | **0.73** |
| RNN-LSTM | 0.79 | 0.39 | 0.79 | 0.66 |

**Key insight:** Despite class-weighted training, RNN-LSTM achieves inferior neutral class performance (F1: 0.39) compared to SVM (F1: 0.47) without specialized techniques.

### Noise-Level Performance
| Noise Level | SVM | RNN-LSTM | Gap |
|-------------|-----|----------|-----|
| Clean | 85.8% | 78.4% | 7.4 pts |
| Low | 75.5% | 61.3% | 14.2 pts |
| Medium | 77.2% | 61.5% | 15.7 pts |
| High | 80.9% | 74.9% | 6.0 pts |

**Counter-intuitive finding:** High-noise text is easier to classify than medium-noise text, suggesting emphatic noise (ALL CAPS, repeated characters) amplifies sentiment signals.

### Preprocessing Comparison (Appendix B)
| Model | Minimal | Aggressive | Change |
|-------|---------|------------|--------|
| SVM | 81.1% | 80.7% | -0.4% |
| RNN-LSTM | 70.5% | 64.6% | **-5.9%** |

**Performance gap widened from 10.8 to 16.2 percentage points**, confirming classical robustness is fundamental rather than preprocessing-dependent.

---
### Dependencies
```
numpy==1.24.3
pandas==2.0.2
scikit-learn==1.3.0
tensorflow==2.13.0
keras==2.13.1
matplotlib==3.7.1
seaborn==0.12.2
nltk==3.8.1
torch==2.0.1
spacy==3.6.0
```


## Design

### Data Splitting
- **Training:** 20,037 samples (60%)
- **Validation:** 6,679 samples (20%)
- **Test:** 6,680 samples (20%)
- **Method:** Stratified sampling (maintains class distribution)

### Hyperparameter Tuning
- **Classical models:** Default scikit-learn parameters (no tuning performed)
- **RNN-LSTM:** Grid search over 3 configurations
  - Embedding dimensions: 128
  - Hidden dimensions: 64-96
  - Dropout rates: 0.3-0.4
  - Learning rates: 0.0005-0.001

### Cross-Validation
- Stratified 5-fold cross-validation for model selection
- Final evaluation on held-out test set

---

## Reproducibility

All experiments are fully reproducible:
- **Random seeds:** Fixed at 42 for all experiments
- **Data splits:** Deterministic stratified sampling
- **Model weights:** Saved checkpoints available in `models/`
- **Environment:** Exact dependency versions in `requirements.txt`

### Reproduce Key Results
```python
# Load trained models
import pickle
with open('models/svm_model.pkl', 'rb') as f:
    svm = pickle.load(f)

# Evaluate on test set
from sklearn.metrics import classification_report, f1_score

y_pred = svm.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
# Output: Macro F1: 0.730
```

---

## 📖 Paper

The complete research paper is available in `paper/final_paper.pdf`.

## Future Work

- **Larger datasets:** Test if neural network advantages emerge at 100k+ samples
- **Transformer models:** Evaluate BERT, RoBERTa on noisy customer reviews
- **Alternative balancing:** Test focal loss, SMOTE, and other techniques
- **Cross-domain validation:** Extend to other review domains (restaurants, hotels, products)
- **Multilingual:** Test robustness across languages
- **Noise-aware preprocessing:** Distinguish emphatic vs obscuring noise..

---


## 📧 Contact

**Elif  Akdeniz**  
Universiteit Antwerpen, Faculty of Arts  



## 📊 Project Statistics

- **Lines of code:** ~3,000
- **Models trained:** 4 (+ 3 RNN variants in hyperparameter tuning)
- **Training time:** ~6 hours 
- **Total experiments:** 20+ (including preprocessing comparison)
- **Paper length:** 4 pages (+ 2 appendices)
- **Figures:** 3 main + 3 appendix
- **Tables:** 5 main + 1 appendix


---

*Last updated: January 12, 2026*
