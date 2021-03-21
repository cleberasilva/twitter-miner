import random
import re
import string

from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords, twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


class Processor:
    def __init__(self, language="english"):
        self.stop_words = stopwords.words(language)
        self.classifier = self.train()

    def clear_data(self, tweet_tokens, stop_words):

        cleaned_tokens = []

        # if you want to see all the tags are available at the link below
        # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        for token, tag in pos_tag(tweet_tokens):
            # replaces links with empty string
            token = re.sub(
                "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\\(\\),]|"
                "(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                token,
            )
            # replaces usernames with empty string
            token = re.sub("(@[A-Za-z0-9_]+)", "", token)

            # Nouns
            if tag.startswith("NN"):
                pos = "n"
            # Verbs
            elif tag.startswith("VB"):
                pos = "v"
            else:
                pos = "a"

            # Lemmatization is the process of grouping inflected forms
            # together as a single base form. Example: fruit is the lemma
            # of the words fruits, fruity, fruitful
            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if (
                len(token) > 0
                and token not in string.punctuation
                and token.lower() not in stop_words
            ):
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    def get_all_words(self, cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token

    def get_tweets_for_model(self, cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)

    def train(self):
        positive_tweet_tokens = twitter_samples.tokenized("positive_tweets.json")
        negative_tweet_tokens = twitter_samples.tokenized("negative_tweets.json")

        positive_cleaned_tokens_list = []
        negative_cleaned_tokens_list = []

        for tokens in positive_tweet_tokens:
            positive_cleaned_tokens_list.append(
                self.clear_data(tokens, self.stop_words)
            )

        for tokens in negative_tweet_tokens:
            negative_cleaned_tokens_list.append(
                self.clear_data(tokens, self.stop_words)
            )

        positive_tokens_for_model = self.get_tweets_for_model(
            positive_cleaned_tokens_list
        )
        negative_tokens_for_model = self.get_tweets_for_model(
            negative_cleaned_tokens_list
        )

        positive_dataset = [
            (tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model
        ]

        negative_dataset = [
            (tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model
        ]

        dataset = positive_dataset + negative_dataset

        random.shuffle(dataset)

        return NaiveBayesClassifier.train(dataset)

    def analyze_sentiment(self, tweet):

        tokens = self.clear_data(word_tokenize(tweet), self.stop_words)

        sentiment = self.classifier.classify(dict([token, True] for token in tokens))
        print(sentiment)
        return sentiment
