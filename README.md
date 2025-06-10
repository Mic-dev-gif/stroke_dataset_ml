**Stroke Prediction Analysis 🚨**

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive pipeline for predicting stroke risk using real-world clinical data, with an emphasis on handling severe class imbalance and optimizing clinically relevant metrics.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Pipeline & Workflow](#pipeline--workflow)
4. [Insights](#Insights)
5. [Usage Examples](#usage-examples)
6. [Contributing](#contributing)
7. [License](#license)
8. [Contact](#contact)

---

## 📌 Project Overview

Predicting stroke risk in patients is critical for early intervention. This project:

* Utilizes a real-world **Stroke Prediction** dataset from Kaggle.
* Tackles severe class imbalance (\~4% stroke cases).
* Compares multiple classifiers (e.g., Logistic Regression, XGBoost).

---

## 📊 Dataset

Source: [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

| Feature Category   | Columns                                                         |
| ------------------ | --------------------------------------------------------------- |
| Demographics       | `gender`, `age`                                                 |
| Health Metrics     | `avg_glucose_level`, `bmi`                                      |
| Medical History    | `hypertension`, `heart_disease`                                 |
| Lifestyle & Social | `ever_married`, `work_type`, `residence_type`, `smoking_status` |
| Target             | `stroke` (0 = No, 1 = Yes)                                      |

*After cleaning (dropping missing BMI, removing outliers), 4,909 records remain.*

---

## 🔄 Pipeline & Workflow

1. **Data Loading & Cleaning**

   * Standardize column names
   * Handle missing BMI values
2. **Feature Engineering**

   * Encoding for linear models
   * Label Encoding for tree-based models
3. **Train-Test Split**

   * Stratified (80% train / 20% test)
4. **Scaling**

   * `StandardScaler` for distance-based models
5. **Imbalance Handling**

   * `SMOTE` in an `imblearn.Pipeline` (avoids data leakage)
6. **Model Training & Tuning**

   * KNN, Logistic Regression, Decision Tree, Random Forest, XGBoost
   * `GridSearchCV` optimizing F1 on stroke class
7. **Threshold Tuning**

   * Adjust probability cut-offs for high-recall vs balanced use cases

---

### Hyperparameter & Threshold Tuning

* **GridSearchCV** for optimal hyperparameters & SMOTE sampling ratio
* **Custom thresholds** (e.g., 0.20 vs. 0.50) to target recall or precision based on clinical needs

---

## 📈 Insights

* **Logistic Regression** offers the best balance (highest F1 & ROC AUC).
* **XGBoost at 0.20** excels in recall—ideal for screening scenarios.

---

## 💻 Usage Examples

```python
# Load trained model
import joblib
model = joblib.load('gridsearch/xgb_model.pkl')

# Predict on new data
import pandas as pd
new_data = pd.read_csv('csv_files/sample_patient.csv')
pred_probs = model.predict_proba(new_data)[:, 1]
pred_labels = (pred_probs >= 0.2).astype(int)
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add new analysis"`)
4. Push (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📬 Contact

## Authors

### Mic-dev-gif (Michele Montalvo)  
- **Email:** [michelemontalvo@outlook.com](mailto:michelemontalvo@outlook.com)  
- **GitHub:** [github.com/Mic-dev-gif](https://github.com/Mic-dev-gif)

### Jorge M. M. L. Rodrigues  
- **Email:** [jorgemmlrodrigues@gmail.com](mailto:jorgemmlrodrigues@gmail.com)  
- **GitHub:** [github.com/JorgeMMLRodrigues](https://github.com/JorgeMMLRodrigues)

*Feel free to open issues or discussions!*
