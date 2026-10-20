import streamlit as st
import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# --- Cached Resource Loading ---
# This ensures NLTK data and models load ONCE and stay in memory
@st.cache_resource
def load_resources():
    # Download required NLTK data quietly
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    stop_words = set(stopwords.words('english'))
    port_stemmer = PorterStemmer()

    # Load neural network and tokenizer
    model = load_model('Imdb.keras')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return model, tokenizer, stop_words, port_stemmer


# Initialize resources
try:
    model, tokenizer, stop_words, port_stemmer = load_resources()
except Exception as e:
    st.error(
        f"Error loading model files. Ensure 'Imdb.keras' and 'tokenizer.pkl' are in the same directory. Details: {e}")
    st.stop()


# --- Preprocessing Pipeline ---
def preprocessing(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^\w\s,.!?]', '', text)

    words = text.split()
    processed_words = [port_stemmer.stem(word) for word in words if word not in stop_words]
    return processed_words


def process_review(review, max_length=180):
    review_processed = preprocessing(review)
    review_sequences = tokenizer.texts_to_sequences([review_processed])
    review_final = pad_sequences(review_sequences, maxlen=max_length, padding='post', truncating='post')
    return review_final


# --- UI Layout ---
st.title("Movie Review Sentiment Analyzer")
st.markdown("""
This app utilizes a custom **Deep Learning model (LSTM/CNN)** to analyze the sentiment of movie reviews. 
Type a review below to see if the system classifies it as **Positive** or **Negative**.
""")

st.write("---")

# User Input Text Area
user_review = st.text_area(
    "Enter your movie review here:",
    placeholder="Example: The cinematography was brilliant and the acting was top-notch. Highly recommended!",
    height=150
)

# Predict Button
if st.button("Analyze Sentiment", type="primary"):
    if user_review.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing text patterns..."):
            # Process the raw text input
            processed_input = process_review(user_review)

            # Run model prediction
            prediction = model.predict(processed_input, verbose=0)
            score = prediction[0][0]

            st.write("### Analysis Result:")

            # Display results dynamically based on threshold
            if score > 0.5:
                st.success(f"### 🎉 Positive Sentiment (Confidence: {score:.2%})")
                # st.balloons()
            else:
                st.error(f"### 😞 Negative Sentiment (Confidence: {(1 - score):.2%})")

# --- Sidebar / Portfolio Info ---
st.sidebar.title("About the Developer")
st.sidebar.info("""
**Technology Stack:**
* Python
* TensorFlow / Keras
* NLTK (Natural Language Toolkit)
* Streamlit
""")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 *Tip: Try entering sarcastic or highly polarizing reviews to test the neural network's boundaries.*")
