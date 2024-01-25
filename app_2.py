import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# Display the HTML content
html_content = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Abusive detector!</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }

        header {
            background-color: #3498db;
            color: #fff;
            padding: 10px;
            text-align: center;
        }

        nav {
            background-color: #2ecc71;
            padding: 10px;
            text-align: center;
        }

        nav button {
            background-color: #e74c3c;
            color: #fff;
            border: none;
            padding: 10px 20px;
            margin: 0 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }

        nav button:hover {
            background-color: #c0392b;
        }

        section {
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>

<body>
    <header>
        <h1>check abuse level before posting</h1>
    </header>
    <nav>
        <button onclick="goToHome()">Back to Home</button>
        <button onclick="openSettings()">Settings</button>
        <button onclick="openMenuOptions()">Menu Options</button>
    </nav>
    <script>
        function goToHome() {
            alert("Redirecting to Home");
            // Implement your logic for going back to home
        }

        function openSettings() {
            alert("Opening Settings");
            // Implement your logic for opening settings
        }

        function openMenuOptions() {
            alert("Opening Menu Options");
            // Implement your logic for opening menu options
        }
    </script>
</body>

</html>


"""

# Set up NLTK
nltk.data.path.append("D:\\nltk_data")
stemmer = SnowballStemmer("english")
stopword = set(stopwords.words("english"))
file_path = r"C:\Users\mssms\OneDrive\Desktop\AM\twitter_data.csv"
df = pd.read_csv(file_path)
df['labels'] = df['class'].map({0: 'Tweet contains hateful content', 1: 'Tweet contains offensive language', 2: 'Tweet doesnot have hate or offensive content'})
df['tweet'] = df['tweet'].apply(lambda text: ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(re.sub(r'\[.?\]|\s+|https?://\S+|www\.\S+|<.?>+|[%]|\w*\d\w*', '', str(text).lower())) if word not in stopword]))

# Vectorize features
cv = CountVectorizer()
x = cv.fit_transform(np.array(df["tweet"]))

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, np.array(df["labels"]), test_size=0.33, random_state=42)

# Train the model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


def predict(tweet):
    cleaned_tweet = ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(re.sub(r'\[.?\]|\s+|https?://\S+|www\.\S+|<.?>+|[%]|\w*\d\w*', '', tweet.lower())) if word not in stopword])
    input_data = cv.transform([cleaned_tweet]).toarray()
    prediction = clf.predict(input_data)[0]
    return prediction


# Streamlit App
st.title("Tweet ideas/thoughts below:")

# Input text box for the user to enter a tweet
tweet_input = st.text_area("Enter a tweet:")


# Display the HTML content
st.markdown(html_content, unsafe_allow_html=True)

# Predict button
if st.button("Predict"):
    if tweet_input:
        result = predict(tweet_input)
        st.write(f"Prediction: {result}")
    else:
        st.warning("Please enter a tweet for prediction.")