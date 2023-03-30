"""Fetch tweets from Twitter API."""

import tweepy

client = tweepy.Client(bearer_token='')
QUERY = 'maandamano'

# Get tweets that contain the word maandamano

# -is:retweet means I don't want retweets
# lang:en is asking for the tweets to be in english
tweets = client.search_recent_tweets(
    query=QUERY,
    tweet_fields=['context_annotations', 'created_at'], max_results=100)

# Print tweets
for tweet in tweets.data:
    print(tweet.text)
    if len(tweet.context_annotations) > 0:
        print(tweet.context_annotations)

# Print the number of tweets
print(len(tweets.data))


# Create a dictionary with the tweets
tweets_dict = {}
for tweet in tweets.data:
    tweets_dict[tweet.id] = tweet.text

# Print the dictionary
print(tweets_dict)

# Clean the dict
tweets_dict_clean = {}
for tweet, tweet_values in tweets_dict.items():
    tweets_dict_clean[tweet] = tweets_dict[tweet].replace('', ' ')

# Print the clean dict
print(tweets_dict_clean)

# Create a list with the tweets
tweets_list = []
for tweet, tweet_values in tweets_dict_clean.items():
    tweets_list.append(tweet_values)

# Print the list
print(tweets_list)
