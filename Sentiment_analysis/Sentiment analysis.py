import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


SIA = SentimentIntensityAnalyzer()

def remove_stopwords(text):
    """This function remove the stopwords"""
    # Tokenisieren Sie den Text
    word_tokens = word_tokenize(text)

    # Liste von Stoppwörtern
    stop_words = set(stopwords.words('english'))

    # Entfernen Sie die Stoppwörter
    word_tagged = [word for word in word_tokens if word.lower() not in stop_words]

    # Join the words back into a string
    word_tagged = ' '.join(word_tagged)

    return word_tagged


def build_vader_model(data):
    result = SIA.polarity_scores(data)
    return result


def main():
    twitter_data = pd.read_csv("C:/Users/nomen/Twitter_Data_Analysis/data/twitter_cleaned.csv",
                               encoding='utf-8')

    # Convert 'Text' column to lowercase
    twitter_data['Text'] = twitter_data['Text'].str.lower()

    # Remove stopwords
    twitter_data['Text'] = twitter_data['Text'].apply(remove_stopwords)

    # use VADER for the Sentiment Analysis
    result_vader = {}
    for i, row in tqdm(twitter_data.iterrows(), total=len(twitter_data)):
        try:
            tweet = row['Text']
            tweet_id = row['Id']  # row number
            result_vader[tweet_id] = build_vader_model(tweet)
        except Exception as e:
            print(f"Error at index {i}: {e}")

    vaders_model = pd.DataFrame(result_vader).T
    vaders_model = vaders_model.reset_index().rename(columns={'index': 'Id'})
    final_model = vaders_model.merge(twitter_data, on='Id', how='left')

    # Speichert den bereinigten DataFrame
    final_model.to_csv("C:/Users/nomen/Twitter_Data_Analysis/Sentiment_analysis/twitter_vader.csv",
                        encoding='utf-8', index=False)


if __name__ == '__main__':
        main()