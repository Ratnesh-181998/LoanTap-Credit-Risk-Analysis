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
<img width="2838" height="1619" alt="image" src="https://github.com/user-attachments/assets/08488022-bd2b-4a8e-b3a9-383a744705b4" />
<img width="2860" height="1534" alt="image" src="https://github.com/user-attachments/assets/4099f886-2142-468b-b158-8ebbcd6fc614" />
<img width="2841" height="1538" alt="image" src="https://github.com/user-attachments/assets/4153bc20-685d-47ba-9bf4-254f76d4ffff" />
<img width="2862" height="1511" alt="image" src="https://github.com/user-attachments/assets/f22c89e1-d5e9-486b-905d-2700f530d83d" />

### 2. 🔍 Exploratory Data Analysis (EDA)
- **Univariate Analysis:** Distribution plots for all features.
- **Bivariate Analysis:** Correlation heatmaps and relationship plots with target variable.
- **Interactive Filters:** Slice data by Loan Status, Grade, and Amount.
- **Visualizations:** Dynamic bar charts, pie charts, and box plots.
<img width="2862" height="1617" alt="image" src="https://github.com/user-attachments/assets/2d048122-03dd-4db9-89fd-45025c370eaa" />
<img width="2848" height="1527" alt="image" src="https://github.com/user-attachments/assets/c25059f0-a6cf-42a9-b31a-59aecbf2d1e5" />
<img width="2879" height="1549" alt="image" src="https://github.com/user-attachments/assets/bcb76d54-20a6-4c63-b18a-d5437e5880ee" />
<img width="2389" height="954" alt="image" src="https://github.com/user-attachments/assets/d275c519-3db0-47af-965e-985510ffc2d8" />
<img width="2369" height="1400" alt="image" src="https://github.com/user-attachments/assets/bc50b16a-efc2-498f-8957-0dcb4188426f" />

### 3. 🔧 Preprocessing
- **Automated Pipeline:** Handles missing values, encodes categorical variables, and scales data.
- **SMOTE Balancing:** Addresses class imbalance in the dataset.
- **Feature Engineering:** Creates new features like `Loan_Tenure` and extracts address details.
<img width="2491" height="1392" alt="image" src="https://github.com/user-attachments/assets/a139f7b1-9d74-48d8-90ee-c9941890d357" />
<img width="2390" height="1426" alt="image" src="https://github.com/user-attachments/assets/a8795fc5-60f1-443a-9ba1-8dee2f5c1d03" />

### 4. 🤖 Modeling
- **Model Comparison:** Side-by-side performance metrics for Logistic Regression, Random Forest, XGBoost, etc.
- **Threshold Tuning:** Interactive slider to optimize classification thresholds based on business needs.
- **ROC Curves:** Visual comparison of model performance.
- **Confusion Matrix:** Detailed breakdown of True/False Positives and Negatives.
<img width="2820" height="1541" alt="image" src="https://github.com/user-attachments/assets/aa4730c5-b03b-4dac-8bde-e8ff10a0e3ee" />
<img width="2391" height="1328" alt="image" src="https://github.com/user-attachments/assets/0b320612-8061-4204-8822-e0debd285e1e" />
<img width="2384" height="1209" alt="image" src="https://github.com/user-attachments/assets/31ef2c44-47e9-416c-b80e-43d5ae8af3f6" />
<img width="2416" height="1078" alt="image" src="https://github.com/user-attachments/assets/f8375683-52e6-40ac-a88d-17f758ec3d93" />
<img width="2401" height="1072" alt="image" src="https://github.com/user-attachments/assets/ee4963c1-684d-4307-ad28-1156f3d681e1" />
<img width="2373" height="1387" alt="image" src="https://github.com/user-attachments/assets/3ea54d6d-9fee-453b-b782-cc9128443b0f" />
<img width="2434" height="1194" alt="image" src="https://github.com/user-attachments/assets/00298f73-ebef-4830-80d4-1d732ba586b0" />
<img width="2433" height="1447" alt="image" src="https://github.com/user-attachments/assets/ab3e5a28-8c3e-4334-b8bf-01e0bb02c1a7" />
<img width="2346" height="1353" alt="image" src="https://github.com/user-attachments/assets/6a8fa597-7183-4447-927c-6f07b31b1f8e" />
<img width="2295" height="1408" alt="image" src="https://github.com/user-attachments/assets/2b434060-93d7-49a4-9107-6dc4074bcbb9" />
<img width="2133" height="1274" alt="image" src="https://github.com/user-attachments/assets/3d626fd3-f6ac-4a96-a13b-77749f1ec7f5" />
<img width="2240" height="1312" alt="image" src="https://github.com/user-attachments/assets/24d89d5d-a3f4-4168-abd9-4acddb1505a7" />
<img width="2324" height="1277" alt="image" src="https://github.com/user-attachments/assets/5604ed17-eb26-45d4-8049-afa146f2e21b" />
<img width="2360" height="1162" alt="image" src="https://github.com/user-attachments/assets/b4d5f845-c400-4f7e-9b27-9d2ec8e02678" />
<img width="2378" height="1205" alt="image" src="https://github.com/user-attachments/assets/a5dfe60f-fa65-478c-a3bb-b682e0da8ac6" />
<img width="2301" height="1421" alt="image" src="https://github.com/user-attachments/assets/8927f7d3-5103-4a37-b88e-c2b33d6d07f4" />
<img width="2298" height="1354" alt="image" src="https://github.com/user-attachments/assets/51b23e94-0d23-410d-b905-c0074c994b7f" />
<img width="2280" height="1383" alt="image" src="https://github.com/user-attachments/assets/237ec63d-2c2b-49e6-836f-f8cf3dad5320" />
<img width="2344" height="1327" alt="image" src="https://github.com/user-attachments/assets/006a820f-0cf7-4573-a788-177d45236f86" />
<img width="2061" height="1350" alt="image" src="https://github.com/user-attachments/assets/2555d415-68ad-47e1-a63d-0a9f2d3e9259" />
<img width="2281" height="1429" alt="image" src="https://github.com/user-attachments/assets/01c10ea1-79e3-40d8-9152-83e6ca6205a7" />
<img width="2120" height="1428" alt="image" src="https://github.com/user-attachments/assets/d08141ce-517c-40b1-b481-7f799f810cb4" />
<img width="2358" height="1349" alt="image" src="https://github.com/user-attachments/assets/49791428-5b28-4d89-9e54-e3518bad3022" />

### 5. 💡 Insights
- **Key Findings:** High-risk factors identified (e.g., Grades E-G, High DTI).
- **Loan Predictor:** Interactive simulator to estimate default probability for new applicants.
- **Portfolio Risk:** Visual distribution of risk across loan grades.
<img width="2879" height="1261" alt="image" src="https://github.com/user-attachments/assets/fd85d279-8bc8-4723-a1c6-dfd8c5eef396" />
<img width="2392" height="1243" alt="image" src="https://github.com/user-attachments/assets/2effb4d1-63db-4c4b-8f27-e41ee8a5d6b5" />
<img width="2329" height="1420" alt="image" src="https://github.com/user-attachments/assets/fb38d228-22f5-49ee-af98-09f6c489776c" />
<img width="2344" height="1448" alt="image" src="https://github.com/user-attachments/assets/ea653c06-54e2-4de3-a2b5-0de9794d8a50" />

### 6. 📚 Complete Analysis
- **Case Study Presentation:** A professional, structured walkthrough of the entire project.
- **Problem Statement:** Clear definition of business objectives.
- **Methodology:** Step-by-step explanation of the analytical approach.
- **Business Recommendations:** Strategic advice on pricing, approval criteria, and verification.
<img width="2830" height="1585" alt="image" src="https://github.com/user-attachments/assets/27ce624f-a5ef-461f-a6fc-3b1007206eed" />
<img width="2359" height="1439" alt="image" src="https://github.com/user-attachments/assets/d8895b08-8bb3-432e-ad32-bb291df39622" />
<img width="2400" height="1444" alt="image" src="https://github.com/user-attachments/assets/b581c2f5-bd2c-4592-90d9-59132818f617" />
<img width="2320" height="1430" alt="image" src="https://github.com/user-attachments/assets/2caa79bb-3b1b-43df-921c-48362054492e" />
<img width="2295" height="1467" alt="image" src="https://github.com/user-attachments/assets/c638dc07-e6cd-475f-b8d5-fdd192300785" />
<img width="2288" height="1214" alt="image" src="https://github.com/user-attachments/assets/b3231711-1cff-4812-8c52-8caf1e3c3953" />
<img width="2352" height="1191" alt="image" src="https://github.com/user-attachments/assets/6f5f1bd4-99cd-4728-9f18-09bf636a51eb" />
<img width="1563" height="1410" alt="image" src="https://github.com/user-attachments/assets/016c2b95-45a7-480c-9b83-9f383aaa5a0b" />
<img width="2145" height="1402" alt="image" src="https://github.com/user-attachments/assets/29542c9c-df08-41f8-b694-a12fade83740" />
<img width="2365" height="1445" alt="image" src="https://github.com/user-attachments/assets/c8016bba-3a32-4c27-919e-517ccd4ee75d" />

### 7. 📝 Logs
- **System Logs:** Real-time tracking of application processes and errors for debugging.
<img width="2381" height="1414" alt="image" src="https://github.com/user-attachments/assets/96a381b4-43ae-48df-9ccd-c8a5855317a2" />

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learng, ML models (Logistic Regression, Random Forest, XGBoost), Imbalanced-learn (SMOTE)
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
- 🐛 Issue Tracker: [GitHub Issues](https://github.com/Ratnesh-181998/LoanTap-Credit-Risk-Analysis/issues)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
