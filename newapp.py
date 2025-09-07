import streamlit as st
import pickle
import nltk
import string
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
nltk.download('punkt')
nltk.download('stopwords')


# Initialize stemmer
ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI setup
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="wide")

# CSS
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}
            
.stTextArea textarea {
    background-color: #2b2b3c;
    color: white;
    border-radius: 10px;
}
.result-card {
    padding: 20px;
    border-radius: 12px;
    margin: 10px 0;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.spam { background-color: #ff4c4c; color: white; }
.ham { background-color: #4CAF50; color: white; }
.scrollable-panel {
    max-height: 600px;
    overflow-y: auto;
    padding-right: 10px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.header("⚙️ Model & Dataset Info")
st.sidebar.write("This app classifies text using a **Machine Learning model** trained with TF-IDF.")
st.sidebar.success("**Model:** Naive Bayes / SVM")
st.sidebar.info("**Dataset:** SMS Spam Collection / Custom Data")

# Title
st.markdown(
    """
    <h1 style='text-align: center;'>📩 Smart Spam Email Classifier</h1>
    <h5 style='text-align: center; margin-top: 0px;'>🚀 Detect spam messages with style & animations</h5>
    """,
    unsafe_allow_html=True
)

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Example animation (email flying)
lottie_email = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_jcikwtux.json")

# Layout
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📨 Input & Classification Result")
    user_input = st.text_area("✍️ Enter one or more texts:", height=200)

    if st.button("🔍 Classify"):
        if user_input.strip() != "":
            messages = user_input.split("\n")
            results = []
            for msg in messages:
                if msg.strip() == "":
                    continue
                transformed_text = transform_text(msg)
                X = vectorizer.transform([transformed_text])
                prediction = model.predict(X)[0]

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    confidence = max(probs)
                else:
                    probs = np.array([1])
                    confidence = 1

                results.append((msg, prediction, confidence, probs))

            # Show results
            for idx, (msg, prediction, confidence, _) in enumerate(results):
                label = "🚨 SPAM" if prediction == 1 else "✅ NOT SPAM"
                css_class = "spam" if prediction == 1 else "ham"
                st.markdown(
                    f"<div class='result-card {css_class}'>Message {idx+1}: {label}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("⚠️ Please enter some text to classify!")

    else:
        # Show animation when no input/classification yet
        st_lottie(lottie_email, speed=1, reverse=False, loop=True,
                  quality="low", height=200, key="email")

with right_col:
    if "results" in locals() and len(results) > 0:
        st.subheader("📊 Details & Probability Scores")
        st.markdown("<div class='scrollable-panel'>", unsafe_allow_html=True)
        for idx, (msg, prediction, confidence, probs) in enumerate(results):
            with st.expander(f"Message {idx+1} Details", expanded=True):
                st.write(f"**Confidence:** {confidence*100:.2f}%")

                # Gauges side by side
                gauge_col1, gauge_col2 = st.columns(2)
                for i, p in enumerate(probs):
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=p*100,
                        title={'text': f"{'SPAM' if i==1 else 'NOT SPAM'}", 'font': {'size': 14}},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "#ff4c4c" if i==1 else "#4CAF50"}}
                    ))
                    fig_gauge.update_layout(width=220, height=220,
                                            margin=dict(l=10, r=10, t=30, b=10))

                    if i == 0:
                        gauge_col1.plotly_chart(fig_gauge, use_container_width=True)
                    else:
                        gauge_col2.plotly_chart(fig_gauge, use_container_width=True)

                # Bar chart below gauges
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.bar(["NOT SPAM", "SPAM"], probs, color=["#4CAF50", "#ff4c4c"])
                ax.set_ylabel("Probability", fontsize=6)
                ax.set_title("Class Probability Distribution", fontsize=7)
                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=6)
                st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)
