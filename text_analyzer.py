import re
import nltk
import torch
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

class TextAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
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

# If you need to recreate the joblib file, you can include the following code:
# df = pd.read_csv('all_kindle_review.csv')
# analyzer = TextAnalyzer()
# df['processed_text'] = df['reviewText'].apply(analyzer.preprocess_text)
# X = df['processed_text']
# y = df['processed_text'].apply(analyzer.get_sentiment)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# analyzer.train_sentiment_model(X_train, y_train)
# joblib.dump(analyzer, 'text_analyzer.joblib')
