# 💰 LoanTap Credit Risk Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ola-driver-churn-prediction-machine-learning-mmntzrjjgxbadbxd4.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## 🚀 Project Overview

**LoanTap-Credit-Risk-Analysis** is an advanced AI-powered credit underwriting system designed to assess loan eligibility and minimize Non-Performing Assets (NPA). Built with **Streamlit** and **Python**, this application leverages Machine Learning models (Logistic Regression, Random Forest, XGBoost) to predict loan defaults and provide actionable business insights.

The application serves as a comprehensive dashboard for data scientists and business stakeholders to explore data, train models, and derive strategies for responsible lending.

---
## 🎬 Demo
- **Streamlit Profile** - https://share.streamlit.io/user/ratnesh-181998
- **Project Demo** - https://loantap-credit-risk-analysis-bg4puocgrxddcwfx3pg5bu.streamlit.app/

---
## 📱 Application Modules

The application is structured into 7 comprehensive tabs, each designed for a specific stage of the analysis pipeline:

### 1. 📊 Data Overview
- **Metrics Dashboard:** Real-time counters for Total Loans, Features, Fully Paid vs. Charged Off rates.
- **Interactive Explorer:** Search, filter, and view raw dataset with column selection.
- **Data Quality:** Visual analysis of missing values and data types.
- **Quick Stats:** Instant statistical summary of numerical features.

### 2. 🔍 Exploratory Data Analysis (EDA)
- **Univariate Analysis:** Distribution plots for all features.
- **Bivariate Analysis:** Correlation heatmaps and relationship plots with target variable.
- **Interactive Filters:** Slice data by Loan Status, Grade, and Amount.
- **Visualizations:** Dynamic bar charts, pie charts, and box plots.

### 3. 🔧 Preprocessing
- **Automated Pipeline:** Handles missing values, encodes categorical variables, and scales data.
- **SMOTE Balancing:** Addresses class imbalance in the dataset.
- **Feature Engineering:** Creates new features like `Loan_Tenure` and extracts address details.

### 4. 🤖 Modeling
- **Model Comparison:** Side-by-side performance metrics for Logistic Regression, Random Forest, XGBoost, etc.
- **Threshold Tuning:** Interactive slider to optimize classification thresholds based on business needs.
- **ROC Curves:** Visual comparison of model performance.
- **Confusion Matrix:** Detailed breakdown of True/False Positives and Negatives.

### 5. 💡 Insights
- **Key Findings:** High-risk factors identified (e.g., Grades E-G, High DTI).
- **Loan Predictor:** Interactive simulator to estimate default probability for new applicants.
- **Portfolio Risk:** Visual distribution of risk across loan grades.

### 6. 📚 Complete Analysis
- **Case Study Presentation:** A professional, structured walkthrough of the entire project.
- **Problem Statement:** Clear definition of business objectives.
- **Methodology:** Step-by-step explanation of the analytical approach.
- **Business Recommendations:** Strategic advice on pricing, approval criteria, and verification.

### 7. 📝 Logs
- **System Logs:** Real-time tracking of application processes and errors for debugging.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
- **Deployment:** Streamlit Cloud

---

## 📂 Project Structure

```
LoanTap-Credit-Risk-Analysis/
├── app.py                  # Main Streamlit application
├── Loan_Tap_Final.ipynb    # Original Jupyter Notebook analysis
├── logistic_regression.txt # Dataset (CSV format)
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── LICENSE                 # MIT License
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ratnesh-181998/LoanTap-Credit-Risk-Analysis.git
   cd LoanTap-Credit-Risk-Analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| **Logistic Regression** | 85% | 0.78 | 0.82 | 0.75 |
| **Random Forest** | 89% | 0.82 | 0.86 | 0.79 |
| **XGBoost** | 91% | 0.85 | 0.88 | 0.81 |

*Note: Performance metrics are based on the test dataset.*

---

## 🤝 Contact & Support

Developed by **RATNESH SINGH**

- 📧 **Email:** [rattudacsit2021gate@gmail.com](mailto:rattudacsit2021gate@gmail.com)
- 💼 **LinkedIn:** [Ratnesh Kumar](https://www.linkedin.com/in/ratneshkumar1998/)
- 🐙 **GitHub:** [Ratnesh-181998](https://github.com/Ratnesh-181998)
- 📱 **Phone:** +91-947XXXXX46

### Project Links
- 🌐 Live Demo: [Streamlit](https://loantap-credit-risk-analysis-bg4puocgrxddcwfx3pg5bu.streamlit.app/)
- 📖 Documentation: [GitHub Wiki](https://github.com/Ratnesh-181998/LoanTap-Credit-Risk-Analysis/wiki)
- 🐛 Issue Tracker: [GitHub Issues](https://github.com/LoanTap-Credit-Risk-Analysis/issues)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
