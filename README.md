# Elevate_Lab11

# 🧠 SVM – Breast Cancer Classification (AI & ML Internship Task 11)

## 📌 Objective

The goal of this project is to build, tune, and evaluate a **Support Vector Machine (SVM)** model for **breast cancer classification** using kernel-based learning techniques. This task focuses on understanding margins, kernels, and hyperparameter tuning while following an industry-style ML workflow.

---

## 🛠 Tools & Technologies

* **Python**
* **Scikit-learn**
* **Matplotlib**
* **Google Colab** (execution environment)

---

## 📊 Dataset

* **Primary Dataset:** Sklearn Breast Cancer Dataset
* Loaded using: `sklearn.datasets.load_breast_cancer()`
* Contains 569 samples with 30 numerical features
* Binary classification:

  * `0` → Malignant
  * `1` → Benign

---

## 🚀 Approach / Workflow

1. Load and inspect the breast cancer dataset
2. Split data into training and testing sets
3. Apply **StandardScaler** for feature normalization
4. Build an **SVM Pipeline** (Scaler + SVM)
5. Train SVM using **Linear** and **RBF** kernels
6. Tune hyperparameters (**C** and **gamma**) using **GridSearchCV**
7. Evaluate model performance using:

   * Confusion Matrix
   * Classification Report
8. Plot **ROC Curve** and compute **AUC score**
9. Save:

   * Trained model pipeline
   * Evaluation reports
   * ROC curve image

This structured pipeline-based approach ensures reproducibility and follows real-world ML practices.

---

## 📁 Project Structure

```
SVM-Breast-Cancer-Classification/
│
├── svm_breast_cancer.ipynb
├── README.md
├── requirements.txt
│
├── outputs/
│   ├── roc_curve.png
│   ├── classification_report.txt
│   ├── confusion_matrix.txt
│   ├── auc_score.txt
│   └── best_params.txt
│
├── model/
│   └── svm_pipeline.pkl
```

---

## 📈 Results

* Achieved high classification accuracy using tuned SVM
* ROC–AUC score indicates strong separability between classes
* RBF kernel performed better after hyperparameter tuning

All results are automatically saved to files — **no screenshots required**.

---

## 💾 Saved Artifacts

* **ROC Curve:** `outputs/roc_curve.png`
* **Classification Report:** `outputs/classification_report.txt`
* **Confusion Matrix:** `outputs/confusion_matrix.txt`
* **AUC Score:** `outputs/auc_score.txt`
* **Best Parameters:** `outputs/best_params.txt`
* **Final Model Pipeline:** `model/svm_pipeline.pkl`

---

## ▶ How to Run

```bash
pip install -r requirements.txt
jupyter notebook svm_breast_cancer.ipynb
```

(Or open and run directly in **Google Colab**)

---

## 🎯 Learning Outcome

* Understanding SVM margins and kernels
* Importance of feature scaling in distance-based algorithms
* Practical experience with hyperparameter tuning
* Professional ML workflow using pipelines

---

## 🧠 Interview Questions Covered

* What is margin in SVM?
* Difference between Linear and RBF kernel
* What is C parameter?
* What is gamma?
* Why is scaling required for SVM?

---

## ✅ Final Outcome

This task demonstrates a complete **end-to-end SVM classification workflow** with proper evaluation, tuning, and model persistence — aligned with real-world machine learning practices and internship expectations.

---

📌 *Internship Task 11 – AI & ML Internship*
