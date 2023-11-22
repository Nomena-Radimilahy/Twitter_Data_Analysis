import pandas as pd
import re
import html
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def remove_emojis(text):
    """This funtion removes the emojis and symbols from a given text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed Characters
        "\U0001F004"              # Mahjong Tile Red Dragon
        "\U0001F0CF"              # Playing Card Black Joker
        "\U0001F170-\U0001F171"  # Negative Squared Latin Letters
        "\U0001F17E-\U0001F17F"  # Negative Squared Latin Letters
        "\U0001F18E"              # Negative Squared AB
        "\U0001F191-\U0001F19A"  # Enclosed Characters
        "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)


def remove_urls(text):
    """This function remove the URLs that are in a text"""
    # Use regex to find all web links: http(s):// or www.
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub(r'', text)


def remove_mentions(text):
    """This function removes the user mentions from a text."""
    return re.sub(r'@\w+', '', text)


def remove_hashtags(text):
    """This function removes the hashtags from a text."""
    return re.sub(r'#\w+', '', text)


def to_lower_case(text):
    return text.lower()


def remove_edge_characters(text):
    """
    Entfernt nicht-alphanumerische Zeichen am Anfang und Ende des Textes.
    """
    return re.sub(r'^\W+|\W+$', '', text)


def remove_empty_tweets(df, tweet_column):
    """
    Remove empty or whitespace-only tweets from a list.
    """
    return df[df[tweet_column].str.strip().astype(bool)]


def decode_html_entities(text):
    """Remove HTML entities"""
    return html.unescape(text)


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


def remove_extra_spaces(text):
    """
    Entfernt überflüssige Leerzeichen, Zeilenumbrüche und Tabs aus dem Text.
    """
    text = re.sub(r'\s+', ' ', text)  # Ersetzt jegliche Whitespace-Zeichen durch ein einzelnes Leerzeichen
    text = text.strip()  # Entfernt Leerzeichen am Anfang und am Ende des Textes
    return text


def main():
    twitter_data = pd.read_csv("C:/Users/nomen/Twitter_Data_Analysis/Topic_modeling/discussion.csv", encoding='utf-8')
    tweets = twitter_data['Text']

    print(tweets.iloc[47])

    # remove emojis
    tweets = tweets.apply(remove_emojis)

    # remove URLs
    tweets = tweets.apply(remove_urls)

    # remove mentions
    tweets = tweets.apply(remove_mentions)

    # remove hastags
    tweets = tweets.apply(remove_hashtags)

    # change to lower case
    #tweets = tweets.apply(to_lower_case)

    # remove the quotation marks covering the tweet
    tweets = tweets.apply(remove_edge_characters)

    # remove HTML impurities
    tweets = tweets.apply(decode_html_entities)

    # remove stopwords
    #tweets = tweets.apply(remove_stopwords)

    # remove extra spaces
    tweets = tweets.apply(remove_extra_spaces)

    # Überschreibt die 'Text'-Spalte mit den bereinigten Daten
    twitter_data['Text'] = tweets

    print(tweets.iloc[47])  # corresponds to 49

    # remove empty tweets
    final_data = remove_empty_tweets(twitter_data, 'Text')

    # Speichert den bereinigten DataFrame
    final_data.to_csv("C:/Users/nomen/Twitter_Data_Analysis/Topic_modeling/discussion_cleaned.csv", encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
