"""
app.py

A small Streamlit demo: type a social media-style post, get a live
harmful / not-harmful prediction from the trained Logistic Regression model.

Run with:
    streamlit run app/app.py

Requires that you've already run `python src/train_classical.py` once,
which saves the model + vectorizer into saved_models/.
"""

import sys
import os
import joblib
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import clean_text  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models", "logreg_model.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "saved_models", "tfidf_vectorizer.joblib")

st.set_page_config(page_title="OffensEval Demo", page_icon="🛡️")
st.title("🛡️ OffensEval — Offensive Language Detector")
st.write(
    "Type a short social-media-style post below. The model predicts whether "
    "it would be flagged as harmful/offensive content."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "No trained model found. Run `python src/train_classical.py` first "
        "to train and save the model."
    )
else:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    text_input = st.text_area("Enter a post:", height=100, placeholder="e.g. Have a great day everyone!")

    if st.button("Classify"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(text_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0][1]

            if pred == 1:
                st.error(f"⚠️ Predicted: **Harmful** (confidence: {prob:.1%})")
            else:
                st.success(f"✅ Predicted: **Not Harmful** (confidence: {1 - prob:.1%})")

            st.caption(f"Cleaned text used by the model: `{cleaned}`")

st.divider()
st.caption(
    "Classical TF-IDF + Logistic Regression baseline from the OffensEval project. "
    "See the [full project](https://github.com/shanmukhacharan/OffensEval) for the "
    "BiLSTM and CNN+LSTM deep learning comparison."
)
