from textblob import TextBlob
import streamlit as st

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment=blob.sentiment.polarity

    if sentiment >0:
        return "positive"
    elif sentiment < 0:
        return "negative"
    else:
        return "neutral"
    
st.title("tetsttt")
input=st.text_area("fsdfsd: ")
button=st.button("check")

if button:
    if input:
        sentiment=analyze_sentiment(input)
        st.write("sentiemt: ", sentiment)
    else :
        st.warning("chưa nhập text")