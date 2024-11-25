import streamlit as st
from vietnamese_nlp.analysis import TextAnalysis
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'vietnamese_nlp')))
# Khởi tạo đối tượng phân tích cảm xúc
analyzer = TextAnalysis()

# Giao diện người dùng
st.title("Sentiment Analysis")
input_text = st.text_area("Nhập dòng cần phân tích: ")
button = st.button("Submit")

if button:
    if input_text:
        # Phân tích cảm xúc của văn bản nhập vào
        sentiment_result = analyzer.analyze_text(input_text)

        # Truy xuất kết quả cảm xúc và điểm số từ dictionary
        sentiment = sentiment_result["overall_sentiment"]
        score = sentiment_result["average_polarity"]

        # Hiển thị kết quả
        st.write("Sentiment: ", sentiment)
        st.write("Average Polarity Score: ", score)
    else:
        st.warning("Chưa nhập văn bản.")
