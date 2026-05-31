import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("twitter_training.csv", header=None)

# Assign column names
df.columns = ["ID", "Topic", "Sentiment", "Tweet"]

# Remove missing values
df.dropna(inplace=True)

# Function for sentiment analysis
def analyze_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Create a new sentiment column
df["Predicted_Sentiment"] = df["Tweet"].apply(analyze_sentiment)

# Display first 5 rows
print("First 5 Records:")
print(df.head())

# Count sentiments
sentiment_counts = df["Predicted_Sentiment"].value_counts()

print("\nSentiment Counts:")
print(sentiment_counts)

# Plot bar chart
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Analysis of Social Media Data")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")
plt.grid(axis="y")
plt.show()

# Pie chart
plt.figure(figsize=(6, 6))
sentiment_counts.plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.title("Sentiment Distribution")
plt.ylabel("")
plt.show()