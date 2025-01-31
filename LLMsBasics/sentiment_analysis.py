from transformers import pipeline

# Load the sentiment-analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# Predict the sentiment of a text
sentiment = sentiment_analysis("Maybe I like the movie")

# Print the result
print(sentiment)