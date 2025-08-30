# Multiple-Disease-Prediction-using-ML--GCELT-Final-Year

## 📌 Overview

This project focuses on building a **Multiple Disease Prediction System** using machine learning models to predict the likelihood of **Heart Disease** and **Diabetes** in patients based on their medical history and clinical parameters.

Our goal is to provide a data-driven decision support system for early diagnosis, enabling better prevention and healthcare outcomes.

---

## 👥 Team Members

- Krishnendu Dasgupta
- Urna Nath
- Sagnick Banerjee
- Kuntal Garai
- Labani Marik
- Debabrata Karfa

---

## 📊 Datasets

### 🔹 Heart Disease Dataset

- **Size:** 91,417 records
- **Positive Cases:** 7,813
- **Negative Cases:** 83,604
- **Features Used:** 17

### 🔹 Diabetes Dataset

- **Size:** 100,000 records
- **Positive Cases:** 8,500
- **Negative Cases:** 91,500
- **Features Used:** 8

---

## 🛠️ Tech Stack

- **Language:** Python
- **Frameworks & Libraries:**
  - NumPy → Numerical computations
  - Pandas → Data manipulation & cleaning
  - Matplotlib & Seaborn → Data visualization
  - Scikit-learn → Machine Learning (Classification models, Evaluation metrics)

---

## 🤖 Models & Results

### Heart Disease Prediction

| Model                        | Accuracy |
| ---------------------------- | -------- |
| Support Vector Machine (SVM) | 72.71%   |
| Decision Tree                | 86.09%   |
| Gaussian Naïve Bayes         | 80.34%   |
| Random Forest                | 89.44%   |
| K-Nearest Neighbors          | 74.69%   |
| Logistic Regression          | 79.35%   |

➡️ **Observation:** Random Forest shows the highest accuracy, but Gaussian Naïve Bayes is the most balanced across metrics.

---

### Diabetes Prediction

| Model                        | Accuracy |
| ---------------------------- | -------- |
| Support Vector Machine (SVM) | 88.50%   |
| Decision Tree                | 94.66%   |
| Gaussian Naïve Bayes         | 88.67%   |
| Random Forest                | 95.98%   |
| K-Nearest Neighbors          | 88.54%   |
| Logistic Regression          | 91.29%   |

➡️ **Observation:** Random Forest achieved the best accuracy (95.98%).

---

## 🏆 Key Results

- **Best Model (Diabetes):** Random Forest (95.98% accuracy)
- **Best Model (Heart Disease):** Gaussian Naïve Bayes (most balanced performance across metrics, despite Random Forest’s higher accuracy but bias in negative predictions)

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/KrishnenduDG/final-year-disease-prediction-model-gcelt

   cd multiple-disease-prediction
   ```

2. Install Dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the required notebook.

Alternatively, you can open the desired notebook and run it in [Google Colab](https://colab.research.google.com/).

### 📌 Future Scopes

- Extend prediction to more diseases (e.g., cancer, stroke).

- Develop a web application (Flask/Streamlit) for interactive predictions.

- Integrate with EHR systems for real-world deployment.

- Improve fairness and reduce bias across demographics.
