import os
import re

import numpy as np
import pandas as pd
import tweet_nlp
from dotenv import load_dotenv
from textblob import TextBlob
from tweepy import API, Cursor, OAuthHandler, Stream, StreamListener

load_dotenv()


class TwitterClient:
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(
            self.twitter_client.user_timeline, id=self.twitter_user
        ).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(
            num_friends
        ):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self, num_tweets):
        home_tweets = []
        for tweet in Cursor(
            self.twitter_client.home_timeline, id=self.twitter_user
        ).items(num_tweets):
            home_tweets.append(tweet)
        return home_tweets


class TwitterAuthenticator:
    def authenticate(self):
        auth = OAuthHandler(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))
        auth.set_access_token(
            os.getenv("ACCESS_TOKEN"), os.getenv("ACCESS_TOKEN_SECRET")
        )
        return auth


class TwitterStreamer:
    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()

    def stream_tweets(self, tweets_filename, hashtag_list):
        listener = TwitterListener(tweets_filename)

        auth = self.twitter_authenticator.authenticate()
        stream = Stream(auth, listener)

        stream.filter(track=hashtag_list)


class TwitterListener(StreamListener):
    def __init__(self, tweets_filename):
        self.tweets_filename = tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.tweets_filename, "a") as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print(f"""Error on data: {e}""")
        return True

    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs
            return False

        print(status)


class TweetAnalyzer:
    def clean_tweet(self, tweet):
        return " ".join(
            re.sub(
                "(@[A-Za-z0-9]+)|([^0-9A-Za-z \\t])|(\\w+:\\/\\/\\S+)", " ", tweet
            ).split()
        )

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def tweets_to_dataframe(self, tweets):
        df = pd.DataFrame(
            data=[
                (
                    tweet.text,
                    tweet.id,
                    len(tweet.text),
                    tweet.created_at,
                    tweet.source,
                    tweet.favorite_count,
                    tweet.retweet_count,
                )
                for tweet in tweets
            ],
            columns=["tweet", "id", "length", "date", "source", "likes", "retweets"],
        )
        return df


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()

    api = twitter_client.get_twitter_client_api()

    tweets = api.user_timeline(screen_name="pycon", count=100)

    df = tweet_analyzer.tweets_to_dataframe(tweets)

    processor = tweet_nlp.Processor()

    df["sentiment"] = np.array(
        # [tweet_analyzer.analyze_sentiment(tweet) for tweet in df["tweet"]]
        [processor.analyze_sentiment(tweet) for tweet in df["tweet"]]
    )

    print(df.head(10))
