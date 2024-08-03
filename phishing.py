import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Load the dataset
file_path = 'Phishing_Legitimate_full.csv'
df = pd.read_csv(file_path)

# Drop irrelevant columns
df = df.drop(columns=['id'])

# Prepare the data
X = df.drop(columns=['CLASS_LABEL'])
y = df['CLASS_LABEL']

# Convert all categorical data to numeric if necessary
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title("Phishing URL Detection")

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Home", "Data Visualization", "Prediction"]
page = st.sidebar.radio("Go to", pages)

if page == "Home":
    st.header("Home")
    st.image('ph.jpeg')
    st.write("This app helps to detect phishing URLs based on their characteristics.")
    st.write("Here is a preview of the dataset:")
    st.dataframe(df.head())
    st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

elif page == "Data Visualization":
    st.header("Data Visualization")

    st.write("### Pie Chart of CLASS_LABEL")
    fig, ax = plt.subplots()
    df['CLASS_LABEL'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    
    st.write("### Top 10 Features by Importance")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(10)
    fig, ax = plt.subplots()
    top_features.plot(kind='barh', ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.write("### Line Chart of NumDots")
    fig, ax = plt.subplots()
    df['NumDots'].value_counts().sort_index().plot(kind='line', ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.write("### Heatmap of Feature Correlation")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.write("### Histogram of NumDots")
    fig, ax = plt.subplots()
    df['NumDots'].plot(kind='hist', bins=30, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.write("### Scatter Plot of NumDots vs UrlLength")
    fig, ax = plt.subplots()
    df.plot(kind='scatter', x='NumDots', y='UrlLength', alpha=0.5, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

elif page == "Prediction":
    st.header("Prediction")
    
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    st.write("### Predict if a URL is phishing")
    
    # Define feature columns
    feature_columns = X.columns.tolist()
    
    # Collect user input
    user_input = {}
    for column in feature_columns:
        user_input[column] = st.number_input(column, value=0.0)
    
    # Create DataFrame for prediction
    user_input_df = pd.DataFrame([user_input])
    
    # Align user input with training data
    user_input_df = user_input_df.reindex(columns=X.columns, fill_value=0)
    
    # Predict
    if st.button('Predict'):
        prediction = model.predict(user_input_df)[0]
        if prediction == 1:
            st.write("The URL is predicted to be **phishing**.")
        else:
            st.write("The URL is predicted to be **legitimate**.")
