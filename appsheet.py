import streamlit as st
import sys
import os
import pandas as pd
from sklearn.metrics import classification_report

# Thêm đường dẫn thư mục 'vietnamese_nlp' vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'vietnamese_nlp')))

# Import lớp TextAnalysis từ analysis.py
from vietnamese_nlp.analysis import TextAnalysis

# Khởi tạo đối tượng phân tích cảm xúc
analyzer = TextAnalysis()

# Giao diện người dùng Streamlit
st.title("Sentiment Analysis")
# Tải lên file
uploaded_file = st.file_uploader("Tải lên file CSV hoặc Excel để phân tích cảm xúc:", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Đọc file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        st.write("Dữ liệu đã tải lên:")
        st.dataframe(df.head())  # Chỉ hiển thị 5 dòng đầu tiên

        # Kiểm tra nếu cột 'reviewText' hoặc 'tweets' tồn tại
        if 'reviewText' in df.columns or 'text' in df.columns:
            text_column = 'reviewText' if 'reviewText' in df.columns else 'text'

            # Kiểm tra xem cột sentiment có tồn tại không
            if 'sentiment' not in df.columns:
                st.error("File không chứa cột 'sentiment'. Vui lòng đảm bảo cột này tồn tại trong dữ liệu.")
            else:
                # Nút phân tích cảm xúc
                if st.button("Phân tích cảm xúc"):
                    results = []
                    y_true = []  # Nhãn thực tế
                    y_pred= []
                    y_pred_textblob = []  # Nhãn dự đoán từ TextBlob
                    y_pred_vader = []  # Nhãn dự đoán từ VADER
                    y_pred_nltk = []  # Nhãn dự đoán từ NLTK

                    # Phân tích từng dòng
                    for text, sentiment in zip(df[text_column].fillna(''), df['sentiment'].fillna('')):
                        try:
                            result = analyzer.analyze_text(text)
                            results.append({
                                "original_text": text,
                                "sentiment_textblob": result["sentiment_textblob"],
                                "score_textblob": result["score_textblob"],
                                "sentiment_vader": result["sentiment_vader"],
                                "score_vader": result["score_vader"],
                                "sentiment_nltk": result["sentiment_nltk"],
                                "score_nltk": result["score_nltk"],
                                "overall_sentiment": result["overall_sentiment"],
                                "average_polarity": result["average_polarity"]
                            })
                            
                            # Thêm nhãn thực tế và nhãn dự đoán vào danh sách
                            y_true.append(sentiment)  # Nhãn thực tế
                            y_pred_textblob.append(result["sentiment_textblob"])  # Nhãn dự đoán từ TextBlob
                            y_pred_vader.append(result["sentiment_vader"])  # Nhãn dự đoán từ VADER
                            y_pred_nltk.append(result["sentiment_nltk"])  # Nhãn dự đoán từ NLTK
                            y_pred.append(result["overall_sentiment"])  # Nhãn dự đoán từ TextBlob
                        except Exception as e:
                            results.append({
                                "original_text": text,
                                "sentiment_textblob": "Error",
                                "score_textblob": 0,
                                "sentiment_vader": "Error",
                                "score_vader": 0,
                                "sentiment_nltk": "Error",
                                "score_nltk": 0,
                                "overall_sentiment": "Error",
                                "average_polarity": 0
                            })
                            st.error(f"Lỗi khi phân tích văn bản: {text}\nLỗi: {e}")

                    # Tạo DataFrame kết quả
                    results_df = pd.DataFrame(results)
                    st.write("Kết quả phân tích cảm xúc:")
                    st.dataframe(results_df.head())

                    # Tính toán classification_report cho từng thư viện
                    if len(y_true) > 0 and len(y_pred_textblob) > 0:
                        report_ovarral = classification_report(y_true, y_pred, target_names=["Positive", "Negative", "Neutral"])
                        st.write("Classification Report for overrall:")
                        st.text(report_ovarral)

                    # Tính toán classification_report cho từng thư viện
                    if len(y_true) > 0 and len(y_pred_textblob) > 0:
                        report_textblob = classification_report(y_true, y_pred_textblob, target_names=["Positive", "Negative", "Neutral"])
                        st.write("Classification Report for TextBlob:")
                        st.text(report_textblob)
                    
                    if len(y_true) > 0 and len(y_pred_vader) > 0:
                        report_vader = classification_report(y_true, y_pred_vader, target_names=["Positive", "Negative", "Neutral"])
                        st.write("Classification Report for VADER:")
                        st.text(report_vader)
                    
                    if len(y_true) > 0 and len(y_pred_nltk) > 0:
                        report_nltk = classification_report(y_true, y_pred_nltk, target_names=["Positive", "Negative", "Neutral"])
                        st.write("Classification Report for NLTK:")
                        st.text(report_nltk)
                    
                    # Tải xuống kết quả
                    st.download_button(
                        label="Tải xuống kết quả dưới dạng CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )
        else:
            st.error("File không chứa cột 'reviewText' hoặc 'tweets'.")
    except Exception as e:
        st.error(f"Lỗi khi xử lý file: {e}")
else:
    st.info("Vui lòng tải lên file để bắt đầu phân tích.")
