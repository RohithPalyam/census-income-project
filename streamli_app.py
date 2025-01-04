import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit app setup
st.title("Income Classification (>50K Prediction)")

# 1. Load and preprocess the dataset
st.header("1. Load and Preprocess the Data")

# Load dataset
@st.cache
def load_data():
    try:
        data = pd.read_csv("adult.csv")
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Load dataset
adult_df = load_data()

if adult_df is not None:
    # Preprocess dataset
    adult_df.columns = adult_df.columns.str.strip().str.lower().str.replace("-", "_")
    adult_df.replace(" ?", np.nan, inplace=True)
    adult_df.dropna(inplace=True)

    # Convert the target variable into binary classes
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

    # Plot education level distribution
    st.write("Education Levels:")
    sns.countplot(y='education', data=adult_df, order=adult_df['education'].value_counts().index, palette='coolwarm')
    plt.title('Education Levels')
    st.pyplot(plt)

    # Hours per week vs income
    st.write("Hours Per Week vs Income:")
    sns.boxplot(x='income', y='hours_per_week', data=adult_df, palette='Set3')
    plt.title('Hours Per Week vs Income')
    st.pyplot(plt)

    # 3. Feature Encoding and Scaling
    st.header("3. Data Preparation")

    # Encode categorical features
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

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC()
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

            # Confusion Matrix
            sns.heatmap(results[model_name]["Confusion Matrix"], annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
            plt.title(f'{model_name} Confusion Matrix')
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Error training {model_name}: {e}")

    # 5. Model Comparison
    st.header("5. Model Comparison")

    comparisons = []
    for model_name, metrics in results.items():
        precision = metrics["Classification Report"]["1"]["precision"]
        recall = metrics["Classification Report"]["1"]["recall"]
        f1 = metrics["Classification Report"]["1"]["f1-score"]
        comparisons.append((model_name, metrics["Accuracy"], precision, recall, f1))

    st.write("Model Comparison:")
    comparison_df = pd.DataFrame(comparisons, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    st.dataframe(comparison_df)

    if not comparison_df.empty:
        best_model = comparison_df.sort_values(by="Accuracy", ascending=False).iloc[0]
        st.write(f"Best Model: {best_model['Model']} with Accuracy: {best_model['Accuracy']}")
    else:
        st.warning("No models were successfully trained.")
else:
    st.error("Unable to load or process the dataset.")
