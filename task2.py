

import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# (tokenization + stopword removal + lemmatization)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    tokens = text.split()
    tokens = [word.strip(string.punctuation) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# TF-IDF Vectorizer with Preprocessing
vectorizer = TfidfVectorizer(preprocessor=preprocess_text, stop_words="english", max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
feature_names = vectorizer.get_feature_names_out()

# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)
y_pred_lr = lr_model.predict(X_test_vec)

print("Logistic Regression Performance")
print(classification_report(y_test, y_pred_lr, target_names=["Ham", "Spam"]))

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_vec, y_train)
y_pred_rf = rf_model.predict(X_test_vec)

print("Random Forest Performance")
print(classification_report(y_test, y_pred_rf, target_names=["Ham", "Spam"]))

# Visualize Important Words (from Logistic Regression)
top_n = 20
coefs = lr_model.coef_[0]
top_positive_indices = np.argsort(coefs)[-top_n:]
top_features = feature_names[top_positive_indices]
top_weights = coefs[top_positive_indices]

plt.figure(figsize=(10, 5))
plt.barh(top_features, top_weights, color="green")
plt.xlabel("Coefficient Weight")
plt.title("Top Spam Indicator Words (Logistic Regression)")
plt.tight_layout()
plt.show()
