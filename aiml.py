import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    'message': [
        'Win money now',
        'Hello how are you',
        'Free entry in contest',
        'Call me now',
        'Congratulations you won'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam']
}

df = pd.DataFrame(data)

# Convert text to numbers
cv = CountVectorizer()
X = cv.fit_transform(df['message'])

# Train model
model = MultinomialNB()
model.fit(X, df['label'])

# Streamlit UI
st.title("Spam Detection App")

msg = st.text_input("Enter Message")

if st.button("Check"):
    data = cv.transform([msg])
    prediction = model.predict(data)
    st.write("Result:", prediction[0])