import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# --- Завантаження NLTK ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Завантаження моделі та векторизатора ---
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# --- Закони ---
law_sets = {
    "🇺🇸 US Law": {
        "18 U.S. Code § 249 (Hate Crimes)": ['kill', 'lynch', 'burn', 'gas', 'slur', 'jew', 'exterminate'],
        "FCC obscenity rules": ['fuck', 'shit', 'bitch', 'nigger', 'faggot']
    },
    "🇪🇺 EU Law": {
        "EU Code of Conduct on Hate Speech": ['hate', 'nazi', 'exterminate', 'terrorist', 'subhuman', 'invader']
    },
    "🇺🇦 Ukrainian Law": {
        "ККУ ст. 161 (\u041fорушення рівноправності)": ['жиди', 'хачі', 'москалі', 'чурки', 'чурка', 'ненавиджу', 'випиляти'],
        "ККУ ст. 300 (Розпалювання ворожнечі)": ['знищити', 'очистити', 'ліквідувати', 'випалити']
    }
}

# --- Переклад ---
def translate_to_english(text):
    translator = Translator()
    try:
        return translator.translate(text, src='auto', dest='en').text
    except:
        return text

# --- Перевірка законів ---
def legal_check_multi(original, translated):
    results = []
    orig = original.lower()
    for law, keywords in law_sets["🇺🇦 Ukrainian Law"].items():
        if any(word in orig for word in keywords):
            results.append(f"🇺🇦 Ukrainian Law – {law}")

    translated = translated.lower()
    for region in ["🇺🇸 US Law", "🇪🇺 EU Law"]:
        for law, keywords in law_sets[region].items():
            if any(word in translated for word in keywords):
                results.append(f"{region} – {law}")

    return results if results else ["No clear legal violation"]

# --- Препроцесинг ---
def preprocess(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Інтерфейс ---
st.title("💬 Hate Speech Detector + Law Checker")
example_tweets = {
    "Hate Speech": "I hate all those people. They should all just disappear.",
    "Offensive Language": "There is nothing wrong with Ariana Grande..... Just cause she don't look like hoe.... Y'all gotta a problem.",
    "Neither": "Wishing everyone a great day and lots of sunshine!",
}
selected = st.selectbox("Оберіть приклад твіта:", [""] + list(example_tweets.keys()), index=0)
if selected:
    user_input = example_tweets[selected]
    st.info(f"Автоматично вставлено приклад ({selected}): {user_input}")
else:
    user_input = st.text_area("Встав твіт або будь-який текст:")

if st.button("🔍 Перевірити"):
    translated = translate_to_english(user_input)
    cleaned = preprocess(translated)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}

    st.write("### 🌐 Переклад (для моделі):")
    st.code(translated)

    st.write("### 🧰 Класифікація:")
    st.success(f"Результат: {label_map[prediction]}")

    st.write("### ⚖️ Відповідність законодавствам:")
    for law in legal_check_multi(user_input, translated):
        st.warning(f"Потенційне порушення: {law}")

    st.write("### 📊 Ймовірності по класах:")
    proba = model.predict_proba(vec)[0]
    fig, ax = plt.subplots()
    ax.bar(label_map.values(), proba, color=['red', 'orange', 'green'])
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    st.write("### 🌥 Word Cloud:")
    wc = WordCloud(width=800, height=400, background_color='white').generate(cleaned)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

    st.write("### 📏 Довжина тексту:")
    length = len(user_input.split())
    fig_len, ax_len = plt.subplots()
    ax_len.bar(["Length"], [length], color='skyblue')
    ax_len.set_ylabel("Кількість слів")
    st.pyplot(fig_len)

    st.write("### ❌ Матриця помилок:")
    y_true = [prediction]
    y_pred = [np.argmax(proba)]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))
    fig_cm, ax_cm = plt.subplots()
    disp.plot(ax=ax_cm, cmap='Blues')
    st.pyplot(fig_cm)
