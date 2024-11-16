import streamlit as st
import vietnamese_nlp
from vietnamese_nlp.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()

st.title("Analysic")
input=st.text_area("Nhập dòng cần phân tích: ")
button=st.button("Submit")

if button:
    if input:
        sentiment, score = analyzer.analyze_sentiment(input)
        st.write("sentiemt: ", sentiment)
    else :
        st.warning("chưa nhập text")