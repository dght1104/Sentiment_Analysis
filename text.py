import os
import pandas as pd
from analysis import TextAnalysis  # Đảm bảo 'analysis.py' chứa lớp TextAnalysis

# Khởi tạo đối tượng phân tích cảm xúc từ 'analysis'
analyzer = TextAnalysis()  # Đối tượng phân tích với tiền xử lý, tokenization và phân tích cảm xúc

# Đọc tệp CSV vào DataFrame
data_path = r"d:\Y4 HK1\Sentiment_Analysis\datasheet\amazon_reviews.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    print(f"Tệp '{data_path}' không tồn tại.")
    exit()

# Kiểm tra xem DataFrame có cột 'reviewText' không
if 'reviewText' in df.columns:
    # Thay thế NaN trong cột 'reviewText' bằng chuỗi rỗng
    df['reviewText'] = df['reviewText'].fillna('')
else:
    print("Cột 'reviewText' không tồn tại trong dữ liệu.")
    exit()

# Tạo danh sách để lưu kết quả phân tích cảm xúc
results = []

# Duyệt qua từng dòng trong cột 'reviewText' và phân tích cảm xúc
for text in df['reviewText']:
    try:
        # Phân tích cảm xúc, token và tiền xử lý từ lớp TextAnalysis
        result = analyzer.analyze_text(text)  # Gọi phương thức analyze_text từ lớp TextAnalysis
        results.append({
            "reviewText": text,
            "cleaned_text": result['cleaned_text'],
            "tokens": result['tokens'],
            "sentiment_vader": result.get('sentiment_vader'),
            "score_vader": result.get('score_vader'),
            "sentiment_textblob": result.get('sentiment_textblob'),
            "score_textblob": result.get('score_textblob'),
            "sentiment_nltk": result.get('sentiment_nltk'),
            "score_nltk": result.get('score_nltk'),
            "overall_sentiment": result.get('overall_sentiment'),
            "average_polarity": result.get('average_polarity')
        })
    except Exception as e:
        print(f"Không thể phân tích cảm xúc cho văn bản: {text}\nLỗi: {e}")

# Tạo DataFrame mới từ kết quả phân tích cảm xúc
results_df = pd.DataFrame(results)

# Đảm bảo thư mục 'data' tồn tại trước khi lưu tệp
output_dir = os.path.join("data")
os.makedirs(output_dir, exist_ok=True)

# Đường dẫn lưu tệp CSV kết quả
output_path = os.path.join(output_dir, "amazon_reviews_with_sentiment.csv")
results_df.to_csv(output_path, index=False)

# In kết quả
print(f"Kết quả phân tích cảm xúc đã được lưu vào {output_path}")

# Đọc lại và hiển thị 5 dòng đầu tiên
df_result = pd.read_csv(output_path)
print("5 dòng đầu tiên của kết quả phân tích cảm xúc:")
print(df_result.head(5))
