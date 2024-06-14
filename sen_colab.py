import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

class TextAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load pre-trained emotion model and tokenizer
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Initialize TF-IDF vectorizer and SVM classifier
        self.tfidf_vectorizer = TfidfVectorizer()
        self.svm = SVC(C=10, degree=2, gamma='scale', kernel='linear')

    def preprocess_text(self, reviewText):
        text = reviewText.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        text = re.sub(r'\d', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    def get_sentiment(self, text):
        sentiment_score = self.sia.polarity_scores(text)
        compound_score = sentiment_score['compound']
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def predict_emotion(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.emotion_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        predicted_class = torch.argmax(scores).item()
        return emotion_labels[predicted_class], scores.tolist()

    def train_sentiment_model(self, X_train, y_train):
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        self.svm.fit(X_train_tfidf, y_train)

    def predict_sentiment(self, text):
        text_tfidf = self.tfidf_vectorizer.transform([text])
        return self.svm.predict(text_tfidf)[0]

    def analyze_text(self, text):
        preprocessed_text = self.preprocess_text(text)
        sentiment = self.get_sentiment(preprocessed_text)
        emotion, emotion_scores = self.predict_emotion(preprocessed_text)
        return sentiment, emotion, emotion_scores

# Load your dataset (assuming a CSV file with 'asin' and 'reviewText' columns)
df = pd.read_csv('all_kindle_review .csv')

# Apply the preprocessing to the 'reviewText' column
analyzer = TextAnalyzer()
df['processed_text'] = df['reviewText'].apply(analyzer.preprocess_text)

# Split the data into training and testing sets
X = df['processed_text']
y = df['processed_text'].apply(analyzer.get_sentiment)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the sentiment model
analyzer.train_sentiment_model(X_train, y_train)

# Save the analyzer using joblib
joblib_file = 'text_analyzer.joblib'
joblib.dump(analyzer, joblib_file)
