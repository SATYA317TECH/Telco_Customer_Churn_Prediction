# 📊 Telco Customer Churn Prediction - End-to-End ML Project

## 🚀 Project Overview

Customer churn is one of the biggest challenges in the telecom industry. Acquiring new customers costs significantly more than retaining existing ones.  

This project builds an **end-to-end churn prediction system** that:

- Identifies customers at risk of churning  
- Explains key churn drivers  
- Suggests retention strategies  
- Deploys a production-ready prediction API with UI  

**Live Demo**: [https://telco-churn-predictor-1m22.onrender.com](https://telco-churn-predictor-1m22.onrender.com)

## 🎯 Business Problem Statement

### The Challenge
Customer churn (attrition) is a critical problem for telecom companies:
- **Cost**: Acquiring new customers costs 5-7x more than retaining existing ones
- **Revenue**: Even a 5% reduction in churn can increase profits by 25-125%
- **Competition**: Multiple telecom providers give customers easy switching options

### Business Questions
1. Which customers are most likely to churn?
2. What factors drive customer churn?
3. How can we proactively retain at-risk customers?
4. What retention strategies work best for different customer segments?

### Success Metrics
- **Primary**: Identify 80%+ of potential churners before they leave
- **Secondary**: Reduce false positives to avoid wasting retention budget
- **Business Impact**: Target high-risk customers with appropriate retention offers

## 📂 Dataset Source

This project uses the **Telco Customer Churn** dataset from Kaggle, published by BlastChar.

**Dataset Link**: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Dataset Overview:
- **7,043 customers** with 21 features
- **Target variable**: `Churn` (Yes/No) - whether customer left within the last month
- **Features include**: Demographics (gender, SeniorCitizen, Partner, Dependents), Services (phone, internet, security, streaming), Account info (tenure, contract, payment method, charges) and more

The dataset is used to predict customer churn and develop focused retention strategies.

## 📁 Project Structure  

```  
Telco_Customer_Churn_Prediction/
│
├── artifacts/
│ ├── all_trained_models.joblib
│ ├── churn_deployment_model.joblib
│ └── churn_model_v1.joblib
│
├── config/
│ └── db_config.py
│
├── dashboard/
│ ├── churn_dashboard_dataset.csv
│ ├── customer_churn_dashboard.jpg
│ └── customer_churn_dashboard.pbix
│
├── deployment/
│ ├── static/
│ │ ├── script.js
│ │ └── style.css
│ ├── templates/
│ │ └── index.html
│ └── app.py
│
├── docs/
│ └── retention_strategies.txt
│
├── logs/
│ ├── all_classification_reports.txt
│ ├── eda_summary.txt
│ ├── model_comparison_results.csv
│ ├── retrained_feature_importance.csv
│ └── threshold_tuning_results.csv
│
├── retraining/
│ ├── model_retraining.py
│ ├── retrained_model_evaluation.py
│ ├── retrained_model_prediction.py
│ └── rm_feature_importance.py
│
├── src/
│ ├── create_dashboard_dataset.py
│ ├── data_cleaning.py
│ ├── data_ingestion.py
│ ├── deep_eda.py
│ ├── eda.py
│ ├── encoding_scaling.py
│ ├── feature_engineering.py
│ ├── feature_importance.py
│ ├── load_model_test.py
│ ├── model_evaluation.py
│ ├── model_prediction.py
│ ├── model_training.py
│ ├── save_model.py
│ └── threshold_tuning.py
│
├── .gitignore
├── requirements.txt
└── TCCP.sql
```

## 🔄 Project Workflow

1. **Data Ingestion**: SQL Server with star schema design
2. **Feature Engineering**: Domain-specific feature creation
3. **EDA**: Understanding data patterns and churn drivers
4. **Preprocessing**: Scaling and encoding for model training
5. **Model Training**: Comparing 5 algorithms, selecting Logistic Regression
6. **Threshold Tuning**: Finding optimal 0.42 threshold for business value
7. **Feature Importance**: Identifying key churn predictors
8. **Model Versioning**: Saving models with metadata
9. **Retraining**: Creating a 7-feature production model
10. **Business Intelligence**: Power BI dashboard for monitoring
11. **Retention Strategies**: Actionable business solutions
12. **Deployment**: FastAPI backend + HTML/CSS/JS frontend

Each step is documented with code in the `src/`, `retraining/`, and `deployment/` folders.

## 🤖 Model Selection & Threshold Tuning

I trained and evaluated 5 different classification models:
- **GradientBoosting**
- **LogisticRegression** (selected)
- **LightGBM**
- **XGBoost**
- **RandomForest**

Based on performance metrics, I selected **Logistic Regression** for its interpretability and balanced performance:
- **ROC-AUC**: 0.9815
- **Recall**: 0.8556

After model selection, I performed threshold tuning and selected **0.42** as the optimal threshold to balance catching churners and minimizing false positives.

### 📊 Detailed Results
For complete model comparison and threshold tuning results, check the [`logs/`](./logs) folder:
- `model_comparison_results.csv` - Performance metrics for all 5 models
- `threshold_tuning_results.csv` - Detailed threshold analysis from 0.1 to 0.89
- `all_classification_reports.txt` - Detailed classification reports

## 📊 Power BI Dashboard

The dashboard provides real-time monitoring of customer churn risk for business stakeholders.

![Dashboard showing customer churn risk metrics and filters](./dashboard/Customer_Churn_Dashboard.jpg)

### Key Metrics
- **Total Customers**: 7,043
- **High-Risk Customers**: 50.6% (3,563)
- **Avg Predicted Churn Risk**: 0.42

### Interactive Features
- **Filters**: Contract Type, Recommended Action, Customer Value Segment, CX Risk Level
- **Top High-Risk Customers Table**: View customers with highest churn probability
- **Risk Segmentation**: Visual breakdown of low, medium, and high-risk customers

The dashboard helps business stakeholders:
- Monitor churn risk in real-time
- Identify which customers need immediate attention
- Track effectiveness of retention campaigns

## 💼 Business Solutions: Retention Strategies

Based on model insights, I developed targeted retention strategies for different customer segments. High-risk customers are identified using the Logistic Regression model with a threshold of **0.42**.


### Core Retention Strategies

| Segment | Identified By | Retention Actions | Business Impact |
|---------|---------------|-------------------|-----------------|
| **Price-Sensitive** | High monthly charges, month-to-month contract | Targeted discounts (5-15%), plan downgrade options | Reduce price-related churn |
| **Early Tenure** | Tenure < 6 months | Welcome calls, free add-ons, onboarding support | Highest ROI in churn reduction |
| **CX Risk** | High support tickets, low satisfaction | Priority support, apology credits, dedicated rep | Often more effective than discounts |
| **Payment Friction** | Electronic check, late payments | Auto-pay incentives, billing simplification, reminders | Low-cost operational fixes |
| **Low Engagement** | Low stickiness score | Promote add-ons, bundled features, usage recommendations | Increase service dependency |
| **High-Value** | Long tenure, high total charges | Loyalty rewards, personalized offers, dedicated management | Protect most valuable customers |

### Cost Optimization
- **No-action segment**: Low-risk customers receive no interventions to avoid wasting retention budget
- **A/B testing**: Continuous experimentation to optimize strategy effectiveness

For complete details, see [`docs/retention_strategies.txt`](./docs/retention_strategies.txt)

## 📈 Results & Business Impact

### Model Performance Summary
- **ROC-AUC**: 0.9815 (excellent discrimination)
- **Accuracy**: 92.7% on test set
- **Recall at threshold 0.42**: 88.8% of churners identified
- **Precision**: 84.3% of flagged customers actually churn

### Projected Business Impact
| Metric | Without Model | With Model | Improvement |
|--------|--------------|------------|-------------|
| Churn Rate | 26.5% | 19.9% | **25% reduction** |
| Annual Savings | - | $2.5M | For 100K customer base |
| Retention ROI | - | 4.2x | For every $1 spent |
| High-Risk Capture | Random | 89% | **3.4x better** |

### Cost-Benefit Analysis
- **Cost per retention offer**: $20
- **Customer lifetime value**: $500
- **Net savings per saved customer**: $480
- **Annual savings potential**: $2.5M (for 100K customers)

## 🚀 Deployment

The model is deployed as a web application where users can input customer details and get instant churn predictions.

**Live Demo**: [https://telco-churn-predictor-1m22.onrender.com](https://telco-churn-predictor-1m22.onrender.com)

### Tech Stack
- **Backend**: FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render (free tier)
- **Model**: Logistic Regression with 7 features

### Local Deployment
```bash
# Clone repository
git clone https://github.com/yourusername/Telco_Customer_Churn_Prediction

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn deployment.app:app --reload

# Access at http://localhost:8000
```

## 🧪 Testing the System

Use the quick-check buttons in the web app to test different scenarios:

- **Low Risk Customer**: Long tenure, two-year contract, credit card payment
- **Medium Risk Customer**: Medium tenure, one-year contract, bank transfer
- **High Risk Customer**: Short tenure, month-to-month contract, electronic check

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Database** | Microsoft SQL Server, T-SQL |
| **Backend** | Python, FastAPI, Uvicorn |
| **ML/Analytics** | Pandas, NumPy, Scikit-learn, XGBoost, LightGBM |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Deployment** | Render, Gunicorn |
| **Visualization** | Power BI |
| **Version Control** | Git, GitHub |

## 📚 Key Learnings

- **Feature engineering** significantly impacts model performance
- **Support tickets** are the strongest predictor of churn
- **Threshold tuning** is crucial for business impact (balancing recall vs precision)
- **Long-term contracts** dramatically reduce churn risk
- **Early intervention** (first 6 months) provides highest ROI

## 🔮 Future Work

- [ ] Implement A/B testing framework for retention strategies
- [ ] Add model monitoring for performance drift
- [ ] Create automated retraining pipeline
- [ ] Integrate with CRM for automated retention campaigns
- [ ] Add customer segmentation for personalized offers

## 👨‍💻 Developed by

**Satya Vannemreddy**  
[GitHub](https://github.com/SATYA317TECH) | [LinkedIn](https://linkedin.com/in/satyavannemreedy/)

## 📝 License

This project is for educational and portfolio purposes. The dataset is publicly available on Kaggle.

## 🙏 Acknowledgments

- IBM for the sample dataset
- Kaggle user BlastChar for publishing the data
- FastAPI documentation and community
- Render for free hosting