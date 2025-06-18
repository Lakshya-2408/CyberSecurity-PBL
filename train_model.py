# train_model.py

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# --- Download NLTK data ---
nltk.download('stopwords')
nltk.download('wordnet')

# --- NLP Preprocessing Setup ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spam_keywords = ['login', 'verify', 'password', 'bank', 'urgent', 'account', 'update', 'click', 'security']

# --- Helper Functions ---
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', ' url ', text)
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        return ' '.join(words)
    return ""

def has_suspicious_url(text):
    trusted_domains = ['forms.gle', 'docs.google.com', 'microsoft.com', 'zoom.us', 'github.com']
    urls = re.findall(r'https?://[^\s]+', text)
    for url in urls:
        if any(trusted in url for trusted in trusted_domains):
            continue
        return 1
    return 0

def spam_keyword_count(text):
    return sum(1 for word in spam_keywords if word in text.lower())

def sentiment_score(text):
    return TextBlob(text).sentiment.polarity  # [-1, 1]

# --- Load and Prepare Data ---
df_phish = pd.read_csv('phishing_email.csv')  # must have 'text_combined' and 'label'
df_legit = pd.read_csv('enron_spam_data.csv')  # must have 'Subject' and 'Message'

df_phish['label'] = 1
df_legit['label'] = 0
df_legit['text_combined'] = df_legit['Subject'].fillna('') + ' ' + df_legit['Message'].fillna('')

df = pd.concat([df_phish[['text_combined', 'label']], df_legit[['text_combined', 'label']]])

# Balance classes via upsampling
df_majority = df[df.label == 1]
df_minority = df[df.label == 0]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1).reset_index(drop=True)

# --- Feature Engineering ---
df['clean_text'] = df['text_combined'].apply(clean_text)
df['suspicious_url'] = df['text_combined'].apply(has_suspicious_url)
df['char_count'] = df['text_combined'].apply(lambda x: len(str(x)))
df['word_count'] = df['text_combined'].apply(lambda x: len(str(x).split()))
df['spam_kw_count'] = df['text_combined'].apply(spam_keyword_count)
df['sentiment'] = df['text_combined'].apply(sentiment_score)

# Simulated Header Features (for model compatibility with app.py)
df['spf_pass'] = df['label'].apply(lambda x: 0 if x == 1 else 1)
df['dkim_pass'] = df['label'].apply(lambda x: 0 if x == 1 else 1)
df['domain_mismatch'] = df['label'].apply(lambda x: 1 if x == 1 else 0)

# --- Text Vectorization ---
tfidf = TfidfVectorizer(max_features=300)
X_text = tfidf.fit_transform(df['clean_text']).toarray()
print("TF-IDF feature count:", X_text.shape[1])

# --- Combine All Features ---
X_other = df[['suspicious_url', 'char_count', 'word_count',
              'spam_kw_count', 'sentiment', 'spf_pass', 'dkim_pass', 'domain_mismatch']].values

X = np.hstack((X_text, X_other))
y = df['label'].values

print("✅ Total features used:", X.shape[1])

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = MultinomialNB()
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Save Model ---
joblib.dump(model, "phishing_model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")
print("✅ Model and vectorizer saved.")
