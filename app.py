from textblob import TextBlob
import streamlit as st
from googletrans import Translator

def analyze_sentiment(text):
    translator = Translator()
    # Dịch văn bản sang tiếng Anh
    translated_text = translator.translate(text, src='vi', dest='en').text
    # Phân tích cảm xúc
    blob = TextBlob(translated_text)
    sentiment = blob.sentiment.polarity  # Lấy điểm cảm xúc

    if sentiment >0:
        return "positive" ,sentiment
    elif sentiment < 0:
        return "negative" ,sentiment
    else:
        return "neutral", sentiment


st.title("tetsttt")
input=st.text_area("fsdfsd: ")
button=st.button("check")

if button:
    if input:
        sentiment=analyze_sentiment(input)
        st.write("sentiemt: ", sentiment)
    else :
        st.warning("chưa nhập text")