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
st.subheader("1.1 Load Dataset")
with st.echo():
    # Load datasets
    adult_df = pd.read_csv("adult.csv")
    
    # Preprocess the dataset
    adult_df.columns = adult_df.columns.str.strip().str.lower().str.replace('-', '_')
    adult_df = adult_df.replace(' ?', np.nan).dropna()

    # Convert target column into binary classes
    adult_df['income'] = adult_df['income'].apply(lambda x: 1 if '>50K' in x else 0)

    st.write("Dataset Head:")
    st.dataframe(adult_df.head())

# Exploratory Data Analysis (EDA)
st.header("2. Exploratory Data Analysis")
st.subheader("2.1 Income Distribution")
with st.echo():
    sns.countplot(x='income', data=adult_df, palette='Set2')
    plt.title('Income Distribution')
    st.pyplot(plt)

st.subheader("2.2 Education Levels Distribution")
with st.echo():
    sns.countplot(y='education', data=adult_df, order=adult_df['education'].value_counts().index, palette='coolwarm')
    plt.title('Education Levels')
    st.pyplot(plt)

st.subheader("2.3 Hours Per Week vs Income")
with st.echo():
    sns.boxplot(x='income', y='hours_per_week', data=adult_df, palette='Set3')
    plt.title('Hours Per Week vs Income')
    st.pyplot(plt)

# Feature Encoding and Scaling
st.header("3. Data Preparation")
st.subheader("3.1 Encoding Categorical Variables")
with st.echo():
    label_encoders = {}
    for column in adult_df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        adult_df[column] = le.fit_transform(adult_df[column])
        label_encoders[column] = le

st.subheader("3.2 Scaling Numeric Features")
with st.echo():
    scaler = StandardScaler()
    numeric_columns = adult_df.select_dtypes(include=['int64', 'float64']).columns
    adult_df[numeric_columns] = scaler.fit_transform(adult_df[numeric_columns])

    st.write("Processed Data Sample:")
    st.dataframe(adult_df.head())

# Model Training and Evaluation
st.header("4. Model Training and Evaluation")
st.subheader("4.1 Train-Test Split")
with st.echo():
    X = adult_df.drop('income', axis=1)
    y = adult_df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader("4.2 Train Models and Evaluate")
with st.echo():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}
    for model_name, model in models.items():
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

# Model Comparison
st.header("5. Model Comparison")
st.subheader("5.1 Compare Model Performance")
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

st.subheader("5.2 Best Model Selection")
with st.echo():
    best_model = comparison_df.sort_values(by="Accuracy", ascending=False).iloc[0]
    st.write(f"Best Model: {best_model['Model']} with Accuracy: {best_model['Accuracy']}")
