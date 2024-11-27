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
        if 'reviewText' in df.columns or 'tweets' in df.columns:
            text_column = 'reviewText' if 'reviewText' in df.columns else 'tweets'

            # Nút phân tích cảm xúc
            if st.button("Phân tích cảm xúc"):
                results = []

                # Phân tích từng dòng
                for text in df[text_column].fillna(''):
                    try:
                        result = analyzer.analyze_text(text)
                        results.append({
                            "original_text": text,
                            "cleaned_text": result['cleaned_text'],
                            "overall_sentiment": result["overall_sentiment"],
                            "average_polarity": result["average_polarity"]
                        })
                    except Exception as e:
                        results.append({
                            "original_text": text,
                            "cleaned_text": "",
                            "overall_sentiment": "Error",
                            "average_polarity": 0
                        })
                        st.error(f"Lỗi khi phân tích văn bản: {text}\nLỗi: {e}")

                # Tạo DataFrame kết quả
                results_df = pd.DataFrame(results)
                st.write("Kết quả phân tích cảm xúc:")
                st.dataframe(results_df)

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