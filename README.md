# Credit Card Default Prediction — Ensemble Machine Learning

## 📌 Project Overview
This project builds a machine learning system to predict **credit card default risk**
using ensemble models. The focus is on **model reliability, interpretability, and
business-aligned evaluation**, rather than raw accuracy.

The final selected model is **Gradient Boosting**, chosen for its superior ability
to identify high-risk default cases.

---

## 📊 Dataset
- **Source:** UCI Credit Card Default Dataset
- **Records:** ~30,000 customers
- **Target Variable:** `default.payment.next.month`
  - `0` → No default
  - `1` → Default

The dataset consists primarily of numerical features describing:
- Credit limits
- Payment history
- Bill amounts
- Payment amounts
- Basic demographic information

---

## 🧠 Models Evaluated
- Logistic Regression (baseline)
- Random Forest
- **Gradient Boosting (final selected model)**

Models were evaluated using:
- ROC AUC
- Precision, Recall, F1-score
- Confusion matrices

Special attention was given to **recall for defaulters**, as false negatives
represent the highest financial risk.

---

## 🏆 Final Model Selection
**Gradient Boosting** was selected as the final model because:
- It captures complex, non-linear risk patterns
- It improves recall for defaulters by focusing on difficult cases
- It outperforms simpler models in risk-sensitive scenarios

Model complexity was carefully controlled to avoid overfitting.

---

## 🔍 Key Insights
- Payment history variables are the strongest predictors of default risk
- Behavioral features dominate over demographic attributes
- Most prediction errors occur near the decision threshold
- Ensemble models significantly improve stability and reliability

---

## 📁 Project Structure
```
credit-default-ensemble-ml/
│
├── data/
│ └── credit_default.csv
│
├── models/
│ └── gbc_model.pkl
│ └── rf_controlled.pkl
│
├── notebooks/
│ ├── day28_data_understanding.ipynb
│ ├── day29_random_forest.ipynb
│ ├── day30_feature_importance.ipynb
│ ├── day31_gradient_boosting.ipynb
│ ├── day32_error_analysis.ipynb
│ └── day33_project_summary.ipynb
│
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## 🚀 Future Improvements
- Probability calibration
- Threshold optimization based on business cost
- Model explainability using SHAP
- Ensemble stacking
- Deployment as a scoring API

---

## 👤 Author
**Abdullah Ashraf**  
Machine Learning Engineer (Portfolio Project)
