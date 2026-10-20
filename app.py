import streamlit as st
import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import requests
from bs4 import BeautifulSoup

# --- Page Configuration & Aesthetics ---
st.set_page_config(
    page_title="🎬 AI Movie Sentiment Analyzer & IMDb Scraper",
    page_icon="✨",
    layout="wide"
)

# Custom CSS for modern styling, cards, badges, and clean typography
st.markdown("""
<style>
    /* Main container and typography */
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1f2937;
        font-family: 'Inter', sans-serif;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%);
        box-shadow: 0 6px 8px -1px rgba(79, 70, 229, 0.3);
    }

    /* Result Boxes */
    .success-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 6px solid #10b981;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-top: 15px;
    }
    .error-box {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 6px solid #ef4444;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-top: 15px;
    }

    /* Card styling */
    .card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
        margin-bottom: 20px;
    }

    /* Badge styling */
    .badge-pos {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neg {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Cached Resource Loading ---
@st.cache_resource
def load_resources():
    """Loads the ML model and preprocessing components once."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.warning(f"NLTK download notice: {e}")

    stop_words = set(stopwords.words('english'))
    port_stemmer = PorterStemmer()

    try:
        model = load_model('Imdb.keras')
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
    except FileNotFoundError as e:
        st.error(f"Critical Error: Model files ('Imdb.keras' or 'tokenizer.pkl') not found. Details: {e}")
        st.stop()

    return model, tokenizer, stop_words, port_stemmer


model, tokenizer, stop_words, port_stemmer = load_resources()


# --- Core ML Functions ---
def preprocessing(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    return [port_stemmer.stem(word) for word in words if word not in stop_words]


def process_review(review, max_length=180):
    review_processed = preprocessing(review)
    review_sequences = tokenizer.texts_to_sequences([review_processed])
    return pad_sequences(review_sequences, maxlen=max_length, padding='post', truncating='post')


# --- Robust IMDb Scraper by Movie Name ---
def search_imdb_movie_reviews(movie_name):
    """Searches IMDb for a movie title, extracts its ID, and scrapes user reviews."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    search_url = f"https://www.imdb.com/find/?q={requests.utils.quote(movie_name)}"

    try:
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title_link = None
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/title/tt' in href and 'externalsite' not in href:
                match = re.search(r'(/title/tt\d+/)', href)
                if match:
                    title_link = match.group(1)
                    break

        if not title_link:
            st.error(f"❌ Could not find an IMDb title matching '{movie_name}'. Try checking the spelling.")
            return []

        reviews_url = f"https://www.imdb.com{title_link}reviews"
        resp = requests.get(reviews_url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup_rev = BeautifulSoup(resp.text, 'html.parser')

        comments = []
        for x in soup_rev.find_all('div', class_=['ipc-html-content-inner-div', 'review-content']):
            text = x.get_text(strip=True)
            if text and text not in comments:
                comments.append(text)

        if not comments:
            for container in soup_rev.find_all('div', class_=['ipc-review-container', 'imdb-user-review']):
                text = container.get_text(separator=" ", strip=True)
                if text and text not in comments:
                    comments.append(text)

        return comments

    except Exception as e:
        st.error(f"❌ Error connecting to IMDb: {e}")
        return []


def display_prediction_result(score):
    if score > 0.5:
        st.markdown(f"""
            <div class='success-box'>
                <h3 style='margin:0; color:#065f46;'>✨ 🎉 POSITIVE SENTIMENT</h3>
                <p style='margin: 8px 0 0 0; color:#047857;'>The deep learning model classifies this review as <b>positive</b> with a confidence score of <b>{score:.2%}</b>.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='error-box'>
                <h3 style='margin:0; color:#991b1b;'>😞 💔 NEGATIVE SENTIMENT</h3>
                <p style='margin: 8px 0 0 0; color:#b91c1c;'>The deep learning model classifies this review as <b>negative</b> with a confidence score of <b>{(1 - score):.2%}</b>.</p>
            </div>
        """, unsafe_allow_html=True)


# =================================================================
# MAIN APPLICATION HEADER & LAYOUT
# =================================================================

st.title("🎬 AI Movie Sentiment Analyzer & IMDb Scraper")
st.markdown("""
<p style='font-size: 1.15rem; color: #4b5563; margin-bottom: 2rem;'>
    Unlock the emotional undertones of any movie review instantly using a custom-trained <b>Deep Learning Neural Network (LSTM/CNN)</b>. Test custom review text, use quick sample reviews, or search live movies directly from IMDb!
</p>
""", unsafe_allow_html=True)

# Tabs for Organized Interface
tab1, tab2, tab3 = st.tabs(["✍️ Manual Review Input", "⚡ Quick Test Examples", "🔎 Live IMDb Movie Search"])

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Analyze Custom Review Text")
    user_review = st.text_area(
        "Type or paste your movie review below:",
        placeholder="Example: An absolute masterpiece! The cinematography was breathtaking and the performances felt deeply authentic.",
        height=180,
        key="custom_review_box"
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyze_btn = st.button("Analyze Review", type="primary")

    if analyze_btn:
        if not user_review.strip():
            st.warning("⚠️ Please provide some review text before analyzing.")
        else:
            with st.spinner("🧠 Running neural network inference..."):
                processed_input = process_review(user_review)
                prediction = model.predict(processed_input, verbose=0)
                display_prediction_result(prediction[0][0])
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚡ Pre-loaded IMDb Test Examples")
    st.markdown(
        "Select a sample IMDb review below to instantly test and evaluate the model's performance without searching manually:")

    sample_reviews = {
        "🌟 Positive Sample 1 (The Shawshank Redemption)": (
            "This is easily one of the greatest films ever made. The story of hope, friendship, "
            "and perseverance through the darkest of circumstances is profoundly moving. Morgan Freeman "
            "and Tim Robbins deliver career-defining performances. An absolute masterpiece from start to finish!"
        ),
        "🌟 Positive Sample 2 (Interstellar)": (
            "A breathtaking sci-fi epic that blends hard science with immense emotional depth. Hans Zimmer's "
            "musical score is nothing short of legendary, vibrating through your soul. Visually stunning and intellectually "
            "stimulating!"
        ),
        "😞 Negative Sample 1 (Generic Blockbuster Flop)": (
            "What a complete waste of two hours. The plot made zero sense, characters had no emotional depth whatsoever, "
            "and the CGI looked like it was rendered on a laptop from 2005. Predictable, boring, and utterly disappointing."
        ),
        "😞 Negative Sample 2 (Disappointing Sequel)": (
            "A painful downgrade from the original. The pacing was completely ruined by unnecessary subplots, lazy writing, "
            "and jokes that completely fell flat. I struggled to stay awake until the credits rolled."
        )
    }

    selected_sample_title = st.selectbox("Choose a test sample:", list(sample_reviews.keys()))
    sample_text = sample_reviews[selected_sample_title]

    st.info(f"**Selected Review Content:**\n\n> {sample_text}")

    if st.button("Evaluate Sample Review", key="eval_sample_btn"):
        with st.spinner("Analyzing sample sentiment..."):
            processed_input = process_review(sample_text)
            prediction = model.predict(processed_input, verbose=0)
            display_prediction_result(prediction[0][0])

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Search IMDb Movie Reviews by Name")
    st.markdown(
        "Type any movie title to fetch live user reviews and evaluate model performance on real-world web data.")

    # Quick suggestion movie badges / clickable shortcuts
    st.markdown("**💡 Popular Movies & TV Show Quick Links to Test:**")
    col_link1, col_link2, col_link3, col_link4, col_link5 = st.columns(5)

    selected_suggestion = None
    with col_link1:
        if st.button("🎬 The Shawshank Redemption"):
            selected_suggestion = "The Shawshank Redemption"
    with col_link2:
        if st.button("🚀 Interstellar"):
            selected_suggestion = "Interstellar"
    with col_link3:
        if st.button("🕶️ Inception"):
            selected_suggestion = "Inception"
    with col_link4:
        if st.button("🦇 The Dark Knight"):
            selected_suggestion = "The Dark Knight"
    with col_link5:
        if st.button("📺 Breaking Bad"):
            selected_suggestion = "Breaking Bad"

    st.markdown("---")

    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        movie_query = st.text_input(
            "Enter Movie or TV Show Name:",
            value=selected_suggestion if selected_suggestion else "",
            placeholder="e.g., Gladiator, Avatar, Oppenheimer",
            key="movie_name_input"
        )
    with col_search2:
        st.write("")  # Spacer
        st.write("")
        search_clicked = st.button("Fetch & Analyze", key="search_movie_btn")

    if search_clicked or selected_suggestion:
        query_to_use = movie_query if movie_query else selected_suggestion
        if not query_to_use.strip():
            st.warning("⚠️ Please enter a movie title.")
        else:
            with st.spinner(f"🔍 Searching IMDb for '{query_to_use}'..."):
                reviews = search_imdb_movie_reviews(query_to_use)

            if reviews:
                st.success(f"Successfully retrieved {len(reviews)} reviews for **{query_to_use}**!")
                st.markdown(
                    f"🔗 [View Official IMDb Page](https://www.imdb.com/find/?q={requests.utils.quote(query_to_use)})")

                # Review selector dropdown
                review_options = {f"Review #{i + 1} (Preview: {rev[:60]}...)": rev for i, rev in enumerate(reviews)}
                selected_key = st.selectbox("Select a review to evaluate:", list(review_options.keys()))
                chosen_review = review_options[selected_key]

                st.info(f"**Full Review Text:**\n\n> {chosen_review}")

                if st.button("Evaluate Selected Review", key="eval_scraped"):
                    with st.spinner("Analyzing sentiment..."):
                        processed_input = process_review(chosen_review)
                        prediction = model.predict(processed_input, verbose=0)
                        display_prediction_result(prediction[0][0])
            else:
                st.info("No reviews could be loaded for this title. Try another popular movie name.")
    st.markdown("</div>", unsafe_allow_html=True)

# =================================================================
# SIDEBAR: PROFESSIONAL DEVELOPER PROFILE (RESUME UPDATED)
# =================================================================

st.sidebar.title("👨‍💻 About the Developer")
st.sidebar.markdown("### **Ahsan Akhtar**")
st.sidebar.markdown("*Machine Learning Engineer | MLOps | NLP & Computer Vision*")
st.sidebar.markdown("---")

# Professional Summary
st.sidebar.subheader("🎯 Profile Summary")
st.sidebar.markdown("""
Machine Learning Engineer with **6+ years of hands-on experience** building end-to-end ML and deep learning solutions across predictive modeling, NLP, computer vision, and MLOps. Strong expertise in Python, PyTorch, TensorFlow, CNNs, LSTM, BERT, MLflow, Docker, FastAPI, CI/CD, and cloud deployments on Azure.
""")

st.sidebar.markdown("---")

# Core Technical Skills
st.sidebar.subheader("🛠️ Technical Expertise")
st.sidebar.markdown("""
* **Programming & Data:** Python, SQL, Pandas, NumPy, EDA, Statistical Analysis
* **Machine Learning:** Predictive Modeling, Model Validation, Hyperparameter Tuning, XGBoost, Random Forest
* **Deep Learning & AI:** TensorFlow, Keras, PyTorch, CNNs, LSTM, BERT, Transfer Learning, Computer Vision, YOLOv8/YOLOv11, RAG, Ollama
* **MLOps & Deployment:** MLflow, Docker, FastAPI, REST APIs, GitHub Actions, CI/CD, Azure, Pytest
* **Visualization & Applications:** Streamlit, Plotly, Matplotlib, Seaborn, Power BI
* **Data Acquisition:** BeautifulSoup, Selenium Web Scraping
""")

st.sidebar.markdown("---")

# Career Highlights & Projects
st.sidebar.subheader("🌟 Experience & Projects")
st.sidebar.markdown("""
* **Freelance ML Engineer (2020–Present):** Built predictive solutions for customer retention (84% AUC), retail forecasting, and NLP sentiment systems.
* **Real-Estate MLOps Pipeline (2025–2026):** Engineered automated data pipelines, MLflow tracking, Docker containerization, FastAPI endpoints, and Azure cloud deployment.
* **Pest Identification System (2022–2023):** Developed a 94.7% accurate CNN system (ResNet50/MobileNetV2) with automated CI/CD.
* **Sentiment Analysis Platform:** Built LSTM-based NLP models with interactive Streamlit analytical dashboards.
""")

st.sidebar.markdown("---")
st.sidebar.caption("🌐 **Open for Remote & Contract Opportunities**")
st.sidebar.markdown(
    "<div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>Built with Passion & Advanced AI.</div>",
    unsafe_allow_html=True)