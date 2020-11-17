import tweepy as tp
from tweepy import OAuthHandler
import pandas as pd
import os
from datetime import date, timedelta



API_SECRET_KEY = "Pdyz5X5ihf7ZxEx1jdoaZQTg6Se2V2SLdOP7TBwVYClk8sWTG1"
API_KEY = "rvlqipJ60zAylMAsuE0TL2UaJ"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAKAlIQEAAAAAzVJpoPy5N0VfRjQseC3OXFXg%2BPM%3D7qQFKGaAWEWzmNiztBazRoSUS3AKmzTT9X70ViSIBJRSd90PyZ"
ACCESS_TOKEN = "48271732-mQRHoHwOHuZlb9ET9YfSW1qVJujGajnmp47JyJEz7"
ACCESS_TOKEN_SECRET = "wxFQLgFwpT3cE8alLtfK5uIXBUKnwaI6wyipL8fAMYDO5"



def get_ids(file):
    links = []
    for line in file:
        tweet_id = line.split('/')[-1]
        links.append(tweet_id.strip())
    return links

def fetch_tweets(ids, df):
    auth = OAuthHandler(API_KEY, API_SECRET_KEY)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tp.API(auth)
    lookups = api.statuses_lookup(ids, tweet_mode="extended")
    for lookup in lookups:
        tweet = {
            "id": lookup.id,
            "username": lookup.user.name,
            "retweetcount": lookup.retweet_count,
            "text": lookup.full_text,
            "created_at": lookup.created_at,
            "likes": lookup.favorite_count,
            "hashtags": lookup.entities["hashtags"],
            "followers": lookup.user.followers_count,
            "location": lookup.place
        }
        df = df.append(tweet, ignore_index=True)
    return df


def main():
    start_date = date(2020, 6, 1)
    end_date = date(2020, 11, 14)
    delta = timedelta(days=1)
    while start_date < end_date:
        os.system('snscrape --max-results 300 twitter-search "(xbox series s OR xbox series x OR xbox OR xsx OR xss) -win lang:en until:' + str(start_date+delta) + 'since:'+str(start_date)+'" > ' + str(start_date) + '.txt')
        file = open(str(start_date)+".txt", "r").readlines()
        ids = get_ids(file)
        df = pd.DataFrame(columns=['id', 'username', 'retweetcount', 'text', 'tweetcreatedts',
                                   'likes', 'hashtags', 'followers', 'location'])
        for i in range(3):
            tweets = fetch_tweets(ids[i*100:(i+1)*100], df)
            tweets.to_csv(f"Tweets-xbox.csv", mode='a', header=False, index = False)

        os.system('snscrape --max-results 300 twitter-search "(Playstation 5 OR PS5 OR Playstation OR dualsense) -win lang:en until:' + str(start_date+delta) + 'since:'+str(start_date)+'" > ' + str(start_date) + '.txt')
        file = open(str(start_date)+".txt", "r").readlines()
        ids = get_ids(file)
        df = pd.DataFrame(columns=['id', 'username', 'retweetcount', 'text', 'tweetcreatedts',
                                   'likes', 'hashtags', 'followers', 'location'])
        for i in range(3):
            tweets = fetch_tweets(ids[i*100:(i+1)*100], df)
            tweets.to_csv(f"Tweets-ps.csv", mode='a', header=False, index = False)
        start_date += delta
    ps = pd.read_csv(f"Tweets-ps.csv")
    xbox = pd.read_csv(f"Tweets-xbox.csv")

    result = pd.concat([ps, xbox])
    final = result.drop_duplicates(subset=['id'])

    final.to_csv(f"Tweets-all.csv", mode='a', header=False, index=False)


if __name__ == '__main__':
    main()