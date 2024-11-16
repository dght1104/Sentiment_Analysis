import re
import pyvi
class TextPreprocessor:
    stop_words = {"là", "của", "và", "những", "được", "có", "trong", "một", "khi", "nếu", "thì"}

    @staticmethod
    def to_lowercase(text):
        return text.lower()

    @staticmethod
    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    @staticmethod
    def remove_extra_whitespace(text):
        return ' '.join(text.split())

    @staticmethod
    def remove_stop_words(text):
        tokens = text.split()
        tokens = [word for word in tokens if word not in TextPreprocessor.stop_words]
        return ' '.join(tokens)

    @classmethod
    def preprocess(cls, text):
        text = cls.to_lowercase(text)
        text = cls.remove_punctuation(text)
        text = cls.remove_extra_whitespace(text)
        text = cls.remove_stop_words(text)
        return text

# text = "Trong khi chúng ta đang học lập trình, có nhiều thách thức nhưng cũng có những niềm vui."
# processed_text = TextPreprocessor.preprocess(text)
# print("Kết quả sau khi xử lý:", processed_text)