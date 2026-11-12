
import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if required
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load trained model and TF-IDF vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text preprocessing (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Streamlit GUI
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter your movie review and find its sentiment.")

movie_name = st.text_input("Movie Name")

review = st.text_area(
    "Write Your Review",
    placeholder="Enter your movie review here..."
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")

    else:
        cleaned_review = clean_text(review)
        review_vector = tfidf.transform([cleaned_review])

        prediction = model.predict(review_vector)[0]
        probabilities = model.predict_proba(review_vector)[0]

        confidence = max(probabilities) * 100

        st.subheader("Prediction Result")

        if prediction == "Positive":
            st.success("😊 Positive Sentiment")

        elif prediction == "Negative":
            st.error("😞 Negative Sentiment")

        else:
            st.info("😐 Neutral Sentiment")

        st.metric("Confidence Score", f"{confidence:.2f}%")

        st.write("### Class-wise Probabilities")

        for sentiment, probability in zip(
            model.classes_, probabilities
        ):
            st.write(f"{sentiment}: {probability * 100:.2f}%")
            st.progress(float(probability))
