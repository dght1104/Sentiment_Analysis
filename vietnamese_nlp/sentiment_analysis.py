from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer as NLTK_VADER 

class SentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression()

    @staticmethod   
    def analyze_sentiment_TextBlob(text):
        # Sử dụng TextBlob để phân tích cảm xúc
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # Trả về điểm số từ -1 đến 1

        # Phân loại cảm xúc dựa trên độ phân cực (polarity)
        if polarity >= 0.1:
            return "Positive", polarity
        elif polarity <=  -0.1:
            return "Negative", polarity
        else:
            return "Neutral", polarity
    
    @staticmethod   
    def analyze_sentiment_Vader(text):
        # Sử dụng vader để phân tích cảm xúc
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound_score = scores["compound"]

        # Phân loại cảm xúc dựa trên compound score của VADER
        if compound_score >= 0.1:
            return "Positive", compound_score
        elif compound_score <= -0.1:
            return "Negative", compound_score
        else:
            return "Neutral", compound_score
        
    @staticmethod   
    def analyze_sentiment_nltk(text):
        # Sử dụng NLTK's SentimentIntensityAnalyzer để phân tích cảm xúc
        nltk_analyzer = NLTK_VADER()
        scores = nltk_analyzer.polarity_scores(text)
        compound_score = scores["compound"]

        # Phân loại cảm xúc dựa trên compound score của VADER
        if compound_score >= 0.1:
            return "Positive", compound_score
        elif compound_score <= -0.1:
            return "Negative", compound_score
        else:
            return "Neutral", compound_score

    @staticmethod
    def analyze_sentiment(text):
        sentiment_textblob, score_textblob = SentimentAnalyzer.analyze_sentiment_TextBlob(text)
        sentiment_vader, score_vader = SentimentAnalyzer.analyze_sentiment_Vader(text)
        sentiment_nltk, score_nltk = SentimentAnalyzer.analyze_sentiment_nltk(text)

        # Tính điểm trung bình từ các điểm số
        average_polarity = (score_textblob + score_vader + score_nltk) / 3

        # Phân loại cảm xúc dựa trên điểm trung bình
        if average_polarity > 0.1:
            return "Positive", average_polarity
        elif average_polarity < -0.1:
            return "Negative", average_polarity
        else:
            return "Neutral", average_polarity
    
# Văn bản mẫu
text = "This item didn't disappoint. It's sturdy, has all the screws included for different flat screen models and yes, it works well."
print(f"Text: {text}")

# Phân tích cảm xúc bằng VADER
test_vader = SentimentAnalyzer.analyze_sentiment_Vader(text)
print(f"VADER Sentiment: {test_vader}")

# Phân tích cảm xúc bằng TextBlob
test_textblob = SentimentAnalyzer.analyze_sentiment_TextBlob(text)
print(f"TextBlob Sentiment: {test_textblob}")

# Phân tích cảm xúc bằng NLTK VADER
test_nltk = SentimentAnalyzer.analyze_sentiment_nltk(text)
print(f"NLTK Sentiment: {test_nltk}")

# Phân tích cảm xúc tổng hợp
text_overall = SentimentAnalyzer.analyze_sentiment(text)  # Gọi trực tiếp từ lớp
print(f"Overall Sentiment: {text_overall}")