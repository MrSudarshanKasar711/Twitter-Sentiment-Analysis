import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import joblib
import re
import nltk

# -------- Load Stopwords ----------
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# -------- Load Model + Tokenizer ----------
MODEL_PATH = r"D:\Sudarshan Kasar\Capstone\Twitter Sentiment Analysis\Twitter sentiment project\notebooks\sentiment_model.h5"
TOKEN_PATH = r"D:\Sudarshan Kasar\Capstone\Twitter Sentiment Analysis\Twitter sentiment project\notebooks\tokenizer.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = joblib.load(TOKEN_PATH)

max_len = 50  # MUST MATCH training

# -------- Cleaning Function ----------
def clean_txt(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# -------- Streamlit UI ----------
st.title("Twitter Sentiment Analysis (LSTM)")
user_input = st.text_area("Enter a Tweet")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned = clean_txt(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        
        pred = model.predict(padded)

        # --- AUTO HANDLE SHAPE ---
        if pred.shape == (1, 1):  # Binary (sigmoid)
            score = pred[0][0]
            if score > 0.5:
                st.success(f"Positive Tweet (score={score:.3f})")
            else:
                st.error(f"Negative Tweet (score={score:.3f})")

        elif pred.shape[1] == 3:  # Multi-class softmax
            labels = ["Negative", "Neutral", "Positive"]
            idx = pred.argmax()
            st.write(f"Prediction: **{labels[idx]}**")
            st.write(f"Probabilities: {pred[0]}")

        else:
            st.error(f"Unexpected model output shape: {pred.shape}")

