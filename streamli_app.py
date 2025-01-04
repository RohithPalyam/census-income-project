import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Streamlit app setup
st.title("Income Classification (>50K Prediction)")

# 1. Load and preprocess the dataset
st.header("1. Load and Preprocess the Data")

# Load dataset
@st.cache
def load_data():
    try:
        data = pd.read_csv("adult.csv")  # Make sure the CSV file is in the correct directory
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Load dataset
adult_df = load_data()

if adult_df is not None:
    # Clean and preprocess the dataset
    adult_df.columns = adult_df.columns.str.strip().str.lower().str.replace("-", "_")
    adult_df.replace(" ?", np.nan, inplace=True)
    adult_df.dropna(inplace=True)  # Drop missing values

    # Convert the target variable into binary classes (0 or 1)
    adult_df["income"] = adult_df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    st.write("Dataset Head:")
    st.dataframe(adult_df.head())

    # 2. Exploratory Data Analysis (EDA)
    st.header("2. Exploratory Data Analysis")

    # Plot income distribution
    st.write("Income Distribution:")
    sns.countplot(x='income', data=adult_df, palette='Set2')
    plt.title('Income Distribution')
    st.pyplot(plt)

    # 3. Feature Encoding and Scaling
    st.header("3. Data Preparation")

    # Encode categorical columns using Label Encoding
    label_encoders = {}
    for column in adult_df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        adult_df[column] = le.fit_transform(adult_df[column])
        label_encoders[column] = le

    # Scale numeric features
    scaler = StandardScaler()
    numeric_columns = adult_df.select_dtypes(include=['int64', 'float64']).columns
    adult_df[numeric_columns] = scaler.fit_transform(adult_df[numeric_columns])

    st.write("Processed Data Sample:")
    st.dataframe(adult_df.head())

    # 4. Model Training and Evaluation
    st.header("4. Model Training and Evaluation")

    # Split data into train and test sets
    X = adult_df.drop('income', axis=1)
    y = adult_df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    try:
        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.write(f"Accuracy: {acc}")
        st.write("Confusion Matrix:")
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
        plt.title('Confusion Matrix')
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error training the model: {e}")

else:
    st.error("Unable to load or process the dataset.")
