# Phishing Detection App
This Streamlit application detects the state of a phishing URL with different features extracted from the URLs (URL Length, IP address, @symbol, favicon icon, Double slash in path, pop window, and IFrame).

An interactive interface for uploading data, visualizing distributions of features, and predicting new URLs is provided.

**Features:**

Home: This is like an app introduction, giving basic information and the user can upload an image relevant to the phishing.

Data Visualization: How different features are distributed in a dataset like URL Length, IP address, @symbol.

Prediction: It will predict whether the URL is Phishing or Legitimate with a machine learning model.

According to the dataset, the prediction was done by the website URL using a Random Forest Classifier with 0.65 model accuracy. 

**Please Follow these Guidelines, before entering the URL of a website**

Length:
URLs with 54 characters or more are more likely to be phishing. Ensure your URL is less than 54 characters if you want to avoid a phishing flag.

@ Symbol:
If the URL contains the @ symbol, it might indicate phishing. Be cautious if you see this symbol in the URL.

IP Address:
URLs that contain an IP address (e.g., 123.456.789.000) are often considered phishing. Avoid using URLs with IP addresses for legitimate purposes.

Double Slashes:
URLs with more than one // in the path (e.g., http://example.com//path) could be phishing. Check your URL for multiple slashes.

Favicon:
URLs containing a favicon in the path (e.g., http://example.com/favicon) might be phishing. Ensure that the favicon is not present in your URL.

Pop-up Windows & Iframes:
While not checked in this application, URLs that use pop-up windows or iframes are generally associated with phishing. Be wary of these features.

Overall:
If any of the above features suggest phishing, the URL will be flagged as phishing. Review all the indicators to determine if the URL might be a phishing attempt.



Example URLs to Test:

Phishing Example: http://123.456.789.000/ (Contains IP Address)

Legitimate Example: https://www.example.com/ (No phishing indicators)
