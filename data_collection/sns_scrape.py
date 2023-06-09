"""We are going to scrape Twitter using snscrape"""

import datetime
import pandas as pd
import snscrape.modules.twitter as sntwitter

# Start
start_date = datetime.date(2023, 3, 20)

# End
end_date = datetime.date(2023, 3, 31)

# QUERY
QUERY = "#MaandamanoMondays OR #MaandamanoThursdays OR #RailaOdinga OR #Azimio"

# Create a list of tweets
tweets_list = []

for i, tweet in enumerate(
    sntwitter.TwitterSearchScraper(
        QUERY +
        ' since:' + 
        start_date.strftime('%Y-%m-%d') +
        ' until:' + 
        end_date.strftime('%Y-%m-%d')
    ).get_items()
):
    if i>1000:
        break
    tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username])

# Create a dataframe from the tweets list above
tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

# Show the first 5 elements of the dataframe
tweets_df.head()

# Save the dataframe as a CSV file
tweets_df.to_csv('data/tweets.csv', sep=',', index = False)
