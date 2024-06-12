from flask import Flask, request, render_template
import joblib
from text_analyzer import TextAnalyzer

# Load the TextAnalyzer model
analyzer = joblib.load('text_analyzer.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        sentiment, emotion, emotion_scores = analyzer.analyze_text(text)
        emotion_scores_dict = dict(zip(["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"], emotion_scores))
        return render_template('index.html', sentiment=sentiment, emotion=emotion, emotion_scores=emotion_scores_dict, text=text)

if __name__ == '__main__':
    app.run(debug=True)
