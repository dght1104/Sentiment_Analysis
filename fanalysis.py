# analysis.py

from vietnamese_nlp.tokenization import VietnameseTokenizer
from vietnamese_nlp.preprocessing import TextPreprocessor
from vietnamese_nlp.sentiment_analysis import SentimentAnalyzer

class TextAnalysis:
    def __init__(self):
        # Khởi tạo các lớp tiền xử lý, phân tách từ và phân tích cảm xúc
        self.preprocessor = TextPreprocessor()
        self.tokenizer = VietnameseTokenizer()
        self.sentiment_analyzer = SentimentAnalyzer()

    def preprocess_text(self, text):
        """Tiền xử lý văn bản"""
        clean_text = self.preprocessor.preprocess(text)
        return clean_text

    def tokenize_text(self, clean_text):
        """Phân tách văn bản thành các từ"""
        tokens = self.tokenizer.tokenize(clean_text)
        return tokens

    def analyze_text(self, text):
        """Phân tích cảm xúc và trả về kết quả"""
        clean_text = self.preprocess_text(text)
        tokens = self.tokenize_text(clean_text)

        # Phân tích cảm xúc
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(' '.join(tokens))

        return {
            "original_text": text,
            "cleaned_text": clean_text,
            "tokens": tokens,
            "sentiment_vader": sentiment_result[0],
            "score_vader": sentiment_result[1],
            "sentiment_textblob": sentiment_result[0],
            "score_textblob": sentiment_result[1],
            "sentiment_nltk": sentiment_result[0],
            "score_nltk": sentiment_result[1],
            "overall_sentiment": sentiment_result[0],
            "average_polarity": sentiment_result[1]
        }


text = "This item didn't disappoint. It's sturdy, has all the screws included for different flat screen models and yes, it works well."
analyzer = TextAnalysis()

# Phân tích văn bản
result = analyzer.analyze_text(text)

    # In kết quả
print("Original Text:", result["original_text"])
print("Cleaned Text:", result["cleaned_text"])
print("Tokens:", result["tokens"])
print("VADER Sentiment:", result["sentiment_vader"], "Score:", result["score_vader"])
print("TextBlob Sentiment:", result["sentiment_textblob"], "Score:", result["score_textblob"])
print("NLTK Sentiment:", result["sentiment_nltk"], "Score:", result["score_nltk"])
print("Overall Sentiment:", result["overall_sentiment"])
print("Average Polarity:", result["average_polarity"])
