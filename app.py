from flask import Flask, request, render_template, jsonify
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pymongo import MongoClient

app = Flask(__name__)

# MongoDB connection setup (adjust the connection string as needed)
client = MongoClient("mongodb://localhost:27017")
db = client["customer_reviews_db"]
collection = db["reviews"]

# Initialize BERT-based sentiment analysis pipeline
bert_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

@app.route("/", methods=["GET"])
def index():
    # Render the homepage with a form for submitting a review.
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    review_text = request.form.get("review", "")
    
    if not review_text.strip():
        return render_template("index.html", error="Please enter a review.")
    
    # Perform sentiment analysis with BERT
    bert_result = bert_sentiment(review_text)[0]
    bert_label = bert_result["label"]
    bert_score = bert_result["score"]
    
    # Perform sentiment analysis with VADER
    vader_scores = vader_analyzer.polarity_scores(review_text)
    
    # Save the review and analysis to MongoDB for future insights
    review_data = {
        "review": review_text,
        "bert": {"label": bert_label, "score": bert_score},
        "vader": vader_scores
    }
    collection.insert_one(review_data)
    
    # Render the page with the analysis results
    return render_template("index.html", 
                           review=review_text, 
                           bert_result=bert_result, 
                           vader_result=vader_scores)

if __name__ == "__main__":
    app.run(debug=True)
