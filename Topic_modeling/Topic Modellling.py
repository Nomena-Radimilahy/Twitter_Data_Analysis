import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def split_separated_words(text):
    """This function removes the separators between the words """
    # Dieses Muster findet Wörter, die mit Bindestrichen oder Unterstrichen verbunden sind
    pattern = r'[\w]+(?:(?:-|_)[\w]+)*'

    # Finde alle Muster im Text
    matches = re.findall(pattern, text)

    # Ersetze Bindestriche und Unterstriche mit Leerzeichen und splitte die Wörter
    split_words = [re.sub(r'[-_]', ' ', match).split() for match in matches]

    # Flache Liste aus der verschachtelten Liste erstellen
    flat_list = [word for sublist in split_words for word in sublist]

    # Die Liste in einen Satz umwandeln
    sentence = ' '.join(flat_list)

    return sentence


def clean_text(text):
    """The function removes punctuations, symbols an numbers """
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Ersetze mehrfache Leerzeichen durch ein einzelnes Leerzeichen
    text = re.sub(r'\s+', ' ', text)

    # Entferne Leerzeichen am Anfang und Ende des Texts
    text = text.strip()

    return text


def lemmatize_words(text):
    """Diese Funktion wendet Lemmatisierung auf alle Wörter im Text an."""
    # Initialisiere den WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Tokenisiere den Text
    word_tokens = word_tokenize(text)

    # Wende Lemmatisierung auf jedes Wort an
    lemmatized_words = [lemmatizer.lemmatize(word, "v") for word in word_tokens]

    # Füge die lemmatisierten Wörter zu einem Satz zusammen
    lemmatized_text = ' '.join(lemmatized_words)

    return lemmatized_text


def remove_short_words(text):
    """The function removes words with a length less than 3"""
    # Split text into words
    exceptions = ["ai", "ki"]
    words = text.split()

    # Keep words with length greater than or equal to 3 or in exceptions list
    long_words = [word for word in words if len(word) >= 3 or word.lower() in exceptions]

    # Join the long words back into a string
    cleaned_text = ' '.join(long_words)

    return cleaned_text


def remove_extra_spaces(text):
    """
    Entfernt überflüssige Leerzeichen, Zeilenumbrüche und Tabs aus dem Text.
    """
    text = re.sub(r'\s+', ' ', text)  # Ersetzt jegliche Whitespace-Zeichen durch ein einzelnes Leerzeichen
    text = text.strip()  # Entfernt Leerzeichen am Anfang und am Ende des Textes
    return text


def remove_empty_tweets(df, tweet_column):
    """
    Remove empty or whitespace-only tweets from a list.
    """
    return df[df[tweet_column].str.strip().astype(bool)]


def main():
    twitter_data = pd.read_csv("C:/Users/nomen/OneDrive/Desktop/Bachelorarbeit/Datenanalyse/discussion_cleaned.csv",
                               encoding='utf-8')
    tweets = twitter_data['Text']

    print(tweets.iloc[1])

    # remove separators
    #tweets = tweets.apply(split_separated_words)

    # remove words length < 3
    #tweets = tweets.apply(remove_short_words)

    # remove symbols and numbers
    #tweets = tweets.apply(clean_text)

    # lemmatize
    #tweets = tweets.apply(lemmatize_words)

    # remove empty spaces
    tweets = tweets.apply(remove_extra_spaces)

    print(tweets.iloc[1])

    # Überschreibt die 'Text'-Spalte mit den bereinigten Daten
    twitter_data['Text'] = tweets

    # leere Tweets entfernen, falls entstanden
    final = remove_empty_tweets(twitter_data, 'Text')

    # Speichert den bereinigten DataFrame
    final.to_csv("C:/Users/nomen/OneDrive/Desktop/Bachelorarbeit/Datenanalyse/python_clean.csv",
                 encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
