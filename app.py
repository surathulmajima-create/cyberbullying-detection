import streamlit as st
import joblib
import re

st.set_page_config(
    page_title="Cyberbullying Detection",
    page_icon="🛡️"
)

@st.cache_resource
def load_model():
    model = joblib.load("cyberbullying_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_model()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

st.title("🛡️ Cyberbullying & Body Shaming Detection")
st.write("Enter a message to detect its category.")

message = st.text_area("Enter your message:")

if st.button("Detect"):
    if message.strip():
        cleaned = clean_text(message)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        st.subheader("Detection Result")

        if prediction == "not_cyberbullying":
            st.success("Non-Cyberbullying")
        elif prediction == "body_shaming":
            st.warning("Body Shaming")
        else:
            st.error("Cyberbullying - " + prediction)

    else:
        st.warning("Please enter a message.")
