import tweepy 
import json
import time
import os
import sys

client = tweepy.Client(bearer_token='')
query = 'maandamano'

# Get tweets that contain the word maandamano

# -is:retweet means I don't want retweets
# lang:en is asking for the tweets to be in english 
tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=100)

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
for tweet in tweets_dict:
    tweets_dict_clean[tweet] = tweets_dict[tweet].replace('', ' ')

# Print the clean dict
print(tweets_dict_clean)

# Create a list with the tweets
tweets_list = []
for tweet in tweets_dict_clean:
    tweets_list.append(tweets_dict_clean[tweet])

# Print the list
print(tweets_list)
