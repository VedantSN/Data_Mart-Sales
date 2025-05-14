import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("Train.csv")

# Fill missing values
df["Item_Weight"].fillna(df["Item_Weight"].mean(), inplace=True)
mode_outlet_size = df.pivot_table(values="Outlet_Size", columns="Outlet_Type", aggfunc=lambda x: x.mode()[0])
missing_values = df["Outlet_Size"].isnull()
df.loc[missing_values, "Outlet_Size"] = df.loc[missing_values, "Outlet_Type"].apply(lambda x: mode_outlet_size[x])

# Normalize fat content
df.replace({"Item_Fat_Content": {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}}, inplace=True)

# Label encoding
le = LabelEncoder()
cols = ["Item_Identifier", "Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]
for col in cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(columns="Item_Outlet_Sales", axis=1)
y = df["Item_Outlet_Sales"]

# Split
tX, X_test, tY, Y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Model
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'enable_categorical': [True]
}

grid = GridSearchCV(estimator=XGBRegressor(tree_method='hist'), param_grid=param_grid, cv=3)

grid.fit(tX, tY)
best_model = grid.best_estimator_

train_preds = best_model.predict(tX)
test_preds = best_model.predict(X_test)
train_r2 = metrics.r2_score(tY, train_preds)
test_r2 = metrics.r2_score(Y_test, test_preds)

# Streamlit UI
st.title("Retail Sales Prediction - Data Mart")

st.write("### Dataset Preview")
st.dataframe(df.head())

st.write("### Feature Distributions")
features_to_plot = ["Item_Weight", "Item_Visibility", "Item_MRP", "Item_Outlet_Sales"]
for feature in features_to_plot:
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax,color='coral')
    st.pyplot(fig)

st.write("### Count Plots")
for feature in ["Outlet_Establishment_Year", "Item_Fat_Content", "Item_Type", "Outlet_Size"]:
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.countplot(x=feature, data=df, ax=ax, palette='Set2')
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.write("### Model Performance")
st.write(f"Train R2 Score: {train_r2:.2f}")
st.write(f"Test R2 Score: {test_r2:.2f}")

# Feature Importance
st.write("### Feature Importance")
feat_importance = pd.Series(best_model.feature_importances_, index=X.columns)
fig, ax = plt.subplots()
feat_importance.nlargest(10).plot(kind='barh', ax=ax, color='mediumseagreen')
st.pyplot(fig)

# Underperforming outlets
st.write("### Underperforming Outlets")
raw_df = pd.read_csv("Train.csv")
under_outlets = raw_df.groupby("Outlet_Identifier")["Item_Outlet_Sales"].sum().sort_values().head()
st.dataframe(under_outlets.reset_index())
# Add option for user to choose prediction method
st.write("### Choose Prediction Method")
prediction_method = st.radio(
    "How would you like to make predictions?",
    ("Upload a CSV file", "Enter values manually")
)

if prediction_method == "Upload a CSV file":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load the uploaded test data
            test_df = pd.read_csv(uploaded_file)
            
            # Store original identifiers for display later
            original_item_ids = test_df["Item_Identifier"].copy()
            original_outlet_ids = test_df["Outlet_Identifier"].copy()
            
            # Fill missing values in test data
            test_df["Item_Weight"].fillna(df["Item_Weight"].mean(), inplace=True)
            
            # Fill missing Outlet_Size
            missing_values_test = test_df["Outlet_Size"].isnull()
            test_df.loc[missing_values_test, "Outlet_Size"] = test_df.loc[missing_values_test, "Outlet_Type"].apply(
                lambda x: mode_outlet_size[x] if x in mode_outlet_size.columns else df["Outlet_Size"].mode()[0]
            )
            
            # Normalize fat content
            test_df.replace({"Item_Fat_Content": {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}}, inplace=True)
            
            # Apply label encoding using a fresh encoder for both training and test data together
            combined_data = pd.concat([df[cols], test_df[cols]], ignore_index=True)
            
            # Convert all values to strings to ensure uniform type
            combined_data[col] = combined_data[col].astype(str)
            temp_encoder = LabelEncoder()
            combined_data[col] = temp_encoder.fit_transform(combined_data[col])
            # Split back into train and test
            test_df[cols] = combined_data.iloc[len(df):][cols]
            
            # Prepare features for prediction
            X_test_final = test_df[X.columns]
            
            X_test_array = X_test_final.values
            
            # Make predictions
            predictions = best_model.predict(X_test_array)
            
            # Creating results DataFrame with original identifiers
            results_df = pd.DataFrame({
                "Item_Identifier": original_item_ids,
                "Outlet_Identifier": original_outlet_ids,
                "Predicted_Item_Outlet_Sales": predictions
            })
            
            # Display results
            st.write("### Test Data Preview with Predictions")
            st.dataframe(results_df.head(10))
            
            # Option to download predictions
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="test_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(f"Error details: {type(e).__name__}")

elif prediction_method == "Enter values manually":
    st.write("### Predict Sales")

    # User inputs for prediction
    input_data = {}
    input_data["Item_Weight"] = st.number_input("Item Weight", min_value=0.0, max_value=30.0, value=10.0)
    input_data["Item_Visibility"] = st.number_input("Item Visibility", min_value=0.0, max_value=0.5, value=0.05)
    input_data["Item_MRP"] = st.number_input("Item MRP", min_value=0.0, max_value=300.0, value=100.0)
    input_data["Outlet_Establishment_Year"] = st.number_input("Outlet Establishment Year", min_value=1900, max_value=2025, value=2000)

    # For categorical features,selectbox with encoded labels
    for col in ["Item_Identifier", "Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]:
        unique_vals = df[col].unique()
        selected_val = st.selectbox(f"Select {col}", options=unique_vals)
        input_data[col] = selected_val

    input_df = pd.DataFrame([input_data])

    # Predict button
    if st.button("Predict Sales"):
        input_df = input_df[X.columns]
        
        # Predict
        prediction = best_model.predict(input_df)
        st.write(f"Predicted Item Outlet Sales: {prediction[0]:.2f}")


