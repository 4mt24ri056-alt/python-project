import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample dataset (you can replace with real dataset)
data = {
    'text': [
        'Win money now', 'Hello friend', 'Free prize claim now',
        'Meeting tomorrow', 'Get rich quick', 'How are you'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# Convert labels to numbers
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.2)

# Vectorization
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

# Model
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))

print("Model trained and saved!")