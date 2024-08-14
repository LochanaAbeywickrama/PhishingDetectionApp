import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import io

# Load the dataset
data_path = "Phishing_Legitimate_full.csv"
df = pd.read_csv(data_path)

# Define feature columns for visualization
feature_columns = [
    'UrlLength', 'AtSymbol', 'IpAddress', 'DoubleSlashInPath', 
    'ExtFavicon', 'PopUpWindow', 'IframeOrFrame'
]

# Function to apply rule-based predictions
def make_predictions(features):
    if features['UrlLength'] >= 54 or \
       features['AtSymbol'] or \
       features['IpAddress'] or \
       features['DoubleSlashInPath'] > 7 or \
       features['ExtFavicon'] or \
       features['PopUpWindow'] or \
       features['IframeOrFrame']:
        return 'Phishing'
    return 'Legitimate'

# Function to extract features from a URL
def extract_features(url):
    features = {}
    features['UrlLength'] = len(url)
    features['AtSymbol'] = int('@' in url)
    features['IpAddress'] = int(any(char.isdigit() for char in url.split('/')[2].split('.')))
    features['DoubleSlashInPath'] = url.count('//')
    features['ExtFavicon'] = int('favicon' in url)
    features['PopUpWindow'] = 0  # Static value for simplicity
    features['IframeOrFrame'] = 0  # Static value for simplicity
    return pd.Series(features)

# Function to predict for a dataset
def predict_dataset(df):
    df['Predicted'] = df.apply(make_predictions, axis=1)
    return df

# Sidebar for navigation
st.sidebar.title("Navigation Pages")
page = st.sidebar.radio("Select", ["Home", "Data Visualization", "Phishing Detection"])

# Home page
if page == "Home":
    st.title("Welcome to the Phishing Detection App")
    st.write("""
        This application allows you to:
        - Detect if a URL is likely to be Phishing or Legitimate using rule-based predictions.
        - Visualize data related to phishing and legitimate URLs.
        - Use a machine learning model to enhance phishing detection.
    """)
    st.image('ph.jpeg')
    st.write("Here is a preview of the dataset:")
    st.dataframe(df.head())
    st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # Media display section
    st.write("### Upload and Display Media relevant to the Phishing")

    # File uploader for images, videos, and audio
    uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg', 'mp4', 'wav', 'mp3'])
    
    if uploaded_file is not None:
        file_type = uploaded_file.type
        
        if file_type.startswith('image/'):
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        elif file_type.startswith('video/'):
            st.video(uploaded_file)
        elif file_type.startswith('audio/'):
            st.audio(uploaded_file)
        else:
            st.write("Unsupported file type.")


# Data Visualization page
elif page == "Data Visualization":
    st.header("Data Visualization")

    # Pie Chart of CLASS_LABEL (assuming it exists)
    if 'CLASS_LABEL' in df.columns:
        # Map labels 1 and 0 to 'Legitimate' and 'Phishing'
        df['CLASS_LABEL'] = df['CLASS_LABEL'].map({1: 'Legitimate', 0: 'Phishing'})

        st.write("### Pie Chart of CLASS_LABEL")
        fig, ax = plt.subplots()
        df['CLASS_LABEL'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')  # Remove y-label to make pie chart clearer
        ax.set_title('Distribution of CLASS_LABEL')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("CLASS_LABEL column not found in dataset.")

    # Histograms for all features
    st.write("### Histograms of Features")
    num_features = len(feature_columns)
    num_rows = (num_features + 2) // 3
    fig, ax = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
    
    for i, feature in enumerate(feature_columns):
        row, col = divmod(i, 3)
        ax[row, col].hist(df[feature].dropna(), bins=30, alpha=0.7)
        ax[row, col].set_title(feature)
        ax[row, col].set_xlabel(feature)
        ax[row, col].set_ylabel('Frequency')
    
    for j in range(num_features, num_rows * 3):
        fig.delaxes(ax.flatten()[j])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Heatmap of Feature Correlation
    st.write("### Heatmap of Feature Correlation")
    fig, ax = plt.subplots()
    correlation_matrix = df[feature_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(fig)
    plt.close(fig)

# Phishing Detection page
elif page == "Phishing Detection":
    st.title("Phishing Detection Predictor")

    st.write("""
        Enter a URL to predict whether it is Legitimate or Phishing using predefined rules or a trained model. 
        You can also train a model using the dataset and see the prediction results.
    """)


    # Prepare the data
    X = df[feature_columns]
    y = df['CLASS_LABEL']  # Assuming this column exists and indicates phishing/legitimate

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Display progress
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    st.markdown(f"<h2 style='color: red;'>**Model Accuracy: {accuracy:.2f}**</h2>", unsafe_allow_html=True)

    st.write()
    st.write()

    st.write("""
        Please follow these guidelines for best results:
    """)

    st.write("""
        **Guidelines for Entering URLs:**
        - **Length:** URLs with a length of 54 characters or more are more likely to be phishing.
        - **`@` Symbol:** If the URL contains the `@` symbol, it could indicate phishing.
        - **IP Address:** URLs containing an IP address are often considered phishing.
        - **Double Slashes:** URLs with more than one `//` in the path may be phishing.
        - **Favicon:** URLs with 'favicon' in the path could be phishing.
        - **Pop-up Windows & Iframes:** These are not currently checked but are generally associated with phishing.
        - **Overall:** If any of these features suggest phishing, the URL will be flagged as phishing.

        **Example URLs to Test:**
        - **Phishing Example:** `http://123.456.789.000/` (Contains IP Address)
        - **Legitimate Example:** `https://www.example.com/` (No phishing indicators)
    """)

    # User input for URL
    url_input = st.text_input("Enter URL")

    # Prediction button
    if st.button("Predict"):
        if url_input:
            # Extract features from the entered URL
            features_df = extract_features(url_input)

            # Predict using the rule-based system
            rule_based_prediction = make_predictions(features_df)
            st.write(f"Rule-Based Prediction: **{rule_based_prediction}**")

            # Predict using the trained model
            model_features_df = pd.DataFrame([features_df])
            model_prediction = model.predict(model_features_df)[0]
            st.write(f"Model-Based Prediction: **{model_prediction}**")

            # Display extracted features (optional for debugging)
            st.write("Extracted Features:")
            st.json(features_df.to_dict())
        else:
            st.write("Please enter a URL.")
