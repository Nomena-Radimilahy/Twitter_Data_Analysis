import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm


SIA = SentimentIntensityAnalyzer()

def build_vader_model(data):
    result = SIA.polarity_scores(data)
    return result


def main():
    twitter_data = pd.read_csv("C:/Users/nomen/Twitter_Data_Analysis/Sentiment_analysis/vader_ready.csv",
                               encoding='utf-8')

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