# vietnamese_nlp/analysis.py
from preprocessing import TextPreprocessor
from tokenization import VietnameseTokenizer
from sentiment_analysis import SentimentAnalyzer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

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

        # Phân tích cảm xúc bằng SentimentAnalyzer
        sentiment_result = SentimentAnalyzer.analyze_sentiment(text)

        return {
            "original_text": text,
            "cleaned_text": clean_text,
            "tokens": tokens,
            "sentiment_textblob": sentiment_result["sentiment_textblob"],
            "score_textblob": sentiment_result["score_textblob"],
            "sentiment_vader": sentiment_result["sentiment_vader"],
            "score_vader": sentiment_result["score_vader"],
            "sentiment_nltk": sentiment_result["sentiment_nltk"],
            "score_nltk": sentiment_result["score_nltk"],
            "overall_sentiment": sentiment_result["overall_sentiment"],
            "average_polarity": sentiment_result["average_polarity"]
        }

# text = "she has been loved Nhi for 5 years but Nhi hasn't loved her"
# analyzer = TextAnalysis()

# # Phân tích văn bản
# result = analyzer.analyze_text(text)

#     # In kết quả
# print("Original Text:", result["original_text"])
# print("Cleaned Text:", result["cleaned_text"])
# print("Tokens:", result["tokens"])
# print("VADER Sentiment:", result["sentiment_vader"], "Score:", result["score_vader"])
# print("TextBlob Sentiment:", result["sentiment_textblob"], "Score:", result["score_textblob"])
# print("NLTK Sentiment:", result["sentiment_nltk"], "Score:", result["score_nltk"])
# print("Overall Sentiment:", result["overall_sentiment"])
# print("Average Polarity:", result["average_polarity"])
