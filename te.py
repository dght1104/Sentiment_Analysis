from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Kiểm tra xem có GPU không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải tokenizer và mô hình PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)

# Đưa mô hình vào device (GPU hoặc CPU)
model.to(device)

# Xử lý câu văn
sentence = "Tôi rất hài lòng với sản phẩm này"
tokens = tokenizer(sentence, return_tensors="pt").to(device)

# Dự đoán cảm xúc
outputs = model(**tokens)

# Lấy kết quả phân loại
predicted_class = torch.argmax(outputs.logits, dim=1).item()

print(f"Giá trị của cảm xúc: {predicted_class}")

# In ra kết quả phân loại
if predicted_class == 0:
    print("Sentiment: Negative")
elif predicted_class == 1:
    print("Sentiment: Neutral")
else:
    print("Sentiment: Positive")