import streamlit as st
import sys
import os
import pandas as pd

# Thêm đường dẫn thư mục 'vietnamese_nlp' vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'vietnamese_nlp')))

# Import lớp TextAnalysis từ analysis.py
from vietnamese_nlp.analysis import TextAnalysis

# Khởi tạo đối tượng phân tích cảm xúc
analyzer = TextAnalysis()

# Giao diện người dùng Streamlit
st.title("Sentiment Analysis")
input_text = st.text_area("Nhập dòng cần phân tích: ")
button = st.button("Submit")

if button:
    if input_text:
        try:
            # Phân tích cảm xúc của văn bản nhập vào
            sentiment_result = analyzer.analyze_text(input_text)

            # Truy xuất kết quả cảm xúc và điểm số từ dictionary
            sentimentTextblob = sentiment_result["sentiment_textblob"]
            sentimentVander = sentiment_result["sentiment_vader"]
            sentimentNltk = sentiment_result["sentiment_nltk"]
            sentiment = sentiment_result["overall_sentiment"]
            score = sentiment_result["average_polarity"]

            # Hiển thị kết quả
            st.write("VADER Sentiment: ", sentimentVander)
            st.write("TextBlod Sentiment: ",  sentimentTextblob )
            st.write("Nltk Sentiment: ", sentimentNltk)
            st.write("Sentiment: ", sentiment)
            st.write("Average Polarity Score: ", score)
        except KeyError as e:
            st.error(f"Lỗi trong quá trình phân tích: Thiếu khóa {e}")
        except Exception as e:
            st.error(f"Lỗi không xác định: {e}")
    else:
        st.warning("Chưa nhập văn bản.")
