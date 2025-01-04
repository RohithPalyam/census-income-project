import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
)

# Streamlit app setup
st.title("Income Classification (>50K Prediction)")

# Load and preprocess the dataset
st.header("1. Load and Preprocess the Data")
with st.echo():
    try:
        # Load datasets
        adult_df = pd.read_csv("adult.csv")

        # Preprocess the dataset
        adult_df.columns = adult_df.columns.str.strip().str.lower().str.replace("-", "_")
        adult_df = adult_df.replace(" ?", np.nan).dropna()

        # Convert target column into binary classes
        adult_df["income"] = adult_df["income"].apply(
            lambda x: 1 if ">50K" in x else 0
        )
        adult_df["income"] = pd.to_numeric(adult_df["income"])  # Ensure numeric

        st.write("Dataset Head:")
        st.dataframe(adult_df.head())
    except Exception as e:
        st.error(f"Error loading or preprocessing dataset: {str(e)}")
        st.stop()

# Exploratory Data Analysis (EDA)
st.header("2. Exploratory Data Analysis")
with st.echo():
    try:
        # Income distribution
        st.write("Income Distribution:")
        sns.countplot(x="income", data=adult_df, palette="Set2")
        plt.title("Income Distribution")
        st.pyplot(plt)

        # Education level distribution
        st.write("Education Levels:")
        sns.countplot(
            y="education",
            data=adult_df,
            order=adult_df["education"].value_counts().index,
            palette="coolwarm",
        )
        plt.title("Education Levels")
        st.pyplot(plt)

        # Hours worked per week vs income
        st.write("Hours Per Week vs Income:")
        sns.boxplot(x="income", y="hours_per_week", data=adult_df, palette="Set3")
        plt.title("Hours Per Week vs Income")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error in EDA: {str(e)}")

# Feature Encoding and Scaling
st.header("3. Data Preparation")
with st.echo():
    try:
        # Encode categorical columns
        label_encoders = {}
        for column in adult_df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            adult_df[column] = le.fit_transform(adult_df[column])
            label_encoders[column] = le

        # Scale numeric features
        scaler = StandardScaler()
        numeric_columns = adult_df.select_dtypes(include=["int64", "float64"]).columns
        adult_df[numeric_columns] = scaler.fit_transform(adult_df[numeric_columns])

        st.write("Processed Data Sample:")
        st.dataframe(adult_df.head())
    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        st.stop()

# Model Training and Evaluation
st.header("4. Model Training and Evaluation")
with st.echo():
    try:
        # Split data into train and test sets
        X = adult_df.drop("income", axis=1)
        y = adult_df["income"]

        # Ensure target values are valid
        if not set(y.unique()).issubset({0, 1}):
            st.error("Target variable contains invalid values. Expected binary classes (0 and 1).")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100),
        }

        results = {}
        for model_name, model in models.items():
            try:
                # Train the model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Evaluate metrics
                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                cr = classification_report(y_test, y_pred, output_dict=True)

                # Store results
                results[model_name] = {
                    "Accuracy": acc,
                    "Confusion Matrix": cm,
                    "Classification Report": cr,
                }

                # Display results
                st.write(f"Model: {model_name}")
                st.write(f"Accuracy: {acc}")
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["<=50K", ">50K"],
                    yticklabels=["<=50K", ">50K"],
                )
                plt.title(f"{model_name} Confusion Matrix")
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
    except Exception as e:
        st.error(f"Error in model training and evaluation: {str(e)}")
        st.stop()

# Model Comparison
st.header("5. Model Comparison")
with st.echo():
    try:
        if results:
            comparisons = []
            for model_name, metrics in results.items():
                precision = metrics["Classification Report"]["1"]["precision"]
                recall = metrics["Classification Report"]["1"]["recall"]
                f1 = metrics["Classification Report"]["1"]["f1-score"]
                comparisons.append(
                    (model_name, metrics["Accuracy"], precision, recall, f1)
                )

            # Convert comparisons to DataFrame
            comparison_df = pd.DataFrame(
                comparisons, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
            )

            # Check if the comparison dataframe is not empty
            if not comparison_df.empty:
                st.write("Model Comparison:")
                st.dataframe(comparison_df)

                # Get the best model
                best_model = comparison_df.sort_values(
                    by="Accuracy", ascending=False
                ).iloc[0]
                st.write(
                    f"Best Model: {best_model['Model']} with Accuracy: {best_model['Accuracy']}"
                )
            else:
                st.error("No valid model results available for comparison.")
        else:
            st.error("No models were successfully trained.")
    except Exception as e:
        st.error(f"Error in model comparison: {str(e)}")
