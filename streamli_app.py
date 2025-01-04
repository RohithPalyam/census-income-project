import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score

# Streamlit app setup
st.title("Income Classification (>50K Prediction)")

# Load and preprocess the dataset
st.header("1. Load and Preprocess the Data")
with st.echo():
    # Load datasets
    try:
        adult_df = pd.read_csv("adult.csv")
    except FileNotFoundError:
        st.error("Dataset file not found. Please upload 'adult.csv'.")
        st.stop()

    # Preprocess the dataset
    adult_df.columns = adult_df.columns.str.strip().str.lower().str.replace('-', '_')
    adult_df = adult_df.replace(' ?', np.nan).dropna()

    # Convert target column into binary classes
    adult_df['income'] = adult_df['income'].apply(lambda x: 1 if '>50K' in x else 0)
    st.write("Dataset Head:")
    st.dataframe(adult_df.head())

# Exploratory Data Analysis (EDA)
st.header("2. Exploratory Data Analysis")
with st.echo():
    st.write("Income Distribution:")
    sns.countplot(x='income', data=adult_df, palette='Set2')
    plt.title('Income Distribution')
    st.pyplot(plt)

# Feature Encoding and Scaling
st.header("3. Data Preparation")
with st.echo():
    # Encode categorical columns
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

# Model Training and Evaluation
st.header("4. Train Models and Evaluate")
with st.echo():
    # Split data into train and test sets
    X = adult_df.drop('income', axis=1)
    y = adult_df['income']
    # Ensure target column is numeric
    y = pd.to_numeric(y, errors='coerce')

    # Check for invalid values
    if y.isnull().any():
        st.error("Target column contains invalid values. Ensure the target column is binary (0 or 1).")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[model_name] = {
                "Accuracy": acc,
                "Confusion Matrix": confusion_matrix(y_test, y_pred),
                "Classification Report": classification_report(y_test, y_pred, output_dict=True)
            }
            st.write(f"Model: {model_name}")
            st.write("Accuracy:", acc)
            sns.heatmap(results[model_name]["Confusion Matrix"], annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
            plt.title(f'{model_name} Confusion Matrix')
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")

# Model Comparison
st.header("5. Model Comparison")
with st.echo():
    comparisons = []
    for model_name, metrics in results.items():
        precision = metrics["Classification Report"]["1"]["precision"]
        recall = metrics["Classification Report"]["1"]["recall"]
        f1 = metrics["Classification Report"]["1"]["f1-score"]
        comparisons.append((model_name, metrics["Accuracy"], precision, recall, f1))
    
    st.write("Model Comparison:")
    comparison_df = pd.DataFrame(comparisons, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    st.dataframe(comparison_df)

    best_model = comparison_df.sort_values(by="Accuracy", ascending=False).iloc[0]
    st.write(f"Best Model: {best_model['Model']} with Accuracy: {best_model['Accuracy']}")
