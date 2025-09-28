import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
stopWords=stopwords.words("english")
from nltk.stem.porter import PorterStemmer
pt=PorterStemmer()

def data_preprocess(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopWords and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(pt.stem(i))

    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
mnb=pickle.load(open('model.pkl','rb'))

st.title("Email spam classifier")

input_sms=st.text_input("enter the message")

if st.button('predict'):
    transformed_sms=data_preprocess(input_sms)

    vector_input=tfidf.transform([transformed_sms])

    result=mnb.predict(vector_input)[0]

    if result==1:
         st.header("Spam")

    else :
       st.header("Non spam")