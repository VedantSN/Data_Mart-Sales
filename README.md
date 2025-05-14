# Data_Mart-Sales Prediction

This Streamlit web app predicts retail item sales using machine learning. It uses the BigMart dataset to build an XGBoost regression model, visualize data, and provide predictions through CSV upload or manual input.

## 🔍 Features
- Interactive Streamlit UI
- Data preprocessing with handling of missing values and label encoding
- XGBoost Regressor with GridSearchCV hyperparameter tuning
- Visualizations: histograms, count plots, and feature importance
- Dual input prediction: upload CSV or enter values manually
- Download option for batch predictions

## 🛠️ How to Run the App

### 1. Clone the repository:
```bash
git clone https://github.com/VedantSN/Data_Mart-Sales.git
cd Data_Mart-Sales
```

### 2. Install required packages:
```bash
pip install -r requirements.txt
```

### 3. Add dataset:
Place the `Train.csv` file in the same directory as the app.

### 4. Run the Streamlit app:
```bash
streamlit run app.py
```
### 5. Prediction Using .CSV:
A sample Test.csv file is provided for prediction using .csv file 

## 📂 Files
- `Data_Mart.py`: Main Streamlit app
- `Train.csv`: Dataset
- `Test.csv`: Sample CSV for prediction
- `requirements.txt`: Python dependencies

## 📊 Model
- Algorithm: XGBoost Regressor
- Performance metric: R² Score
- Optimized using GridSearchCV

## 📜 License
This project is licensed under the MIT License.
