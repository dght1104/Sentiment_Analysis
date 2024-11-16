import os
import pandas as pd
from vietnamese_nlp.sentiment_analysis import SentimentAnalyzer

# Khởi tạo đối tượng phân tích cảm xúc
analyzer = SentimentAnalyzer()

# Đọc tệp CSV vào DataFrame
data_path = r"d:\Y4 HK1\Sentiment_Analysis\datasheet\amazon_reviews.csv"
df = pd.read_csv(data_path)


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
    sentiment, score = analyzer.analyze_sentiment(text)
    results.append({"reviewText": text, "sentiment": sentiment, "score": score})

# Tạo DataFrame mới từ kết quả phân tích cảm xúc
results_df = pd.DataFrame(results)

# Đảm bảo thư mục 'data' tồn tại trước khi lưu tệp
output_dir = os.path.join("datasheet")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Đường dẫn lưu tệp CSV kết quả
output_path = os.path.join(output_dir, "amazon_reviews_with_sentiment.csv")
results_df.to_csv(output_path, index=False)

# In kết quả
print(f"Kết quả phân tích cảm xúc đã được lưu vào {output_path}")

# Đọc lại và hiển thị 5 dòng đầu tiên
df_result = pd.read_csv(output_path)
print("5 dòng đầu tiên của kết quả phân tích cảm xúc:")
print(df_result.head(5))
