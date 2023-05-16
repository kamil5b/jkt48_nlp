import tweepy
import re
import time 
from datetime import datetime,timedelta,timezone
import pandas as pd
import math
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import numpy as np
import threading

import os
from dotenv import load_dotenv 

load_dotenv()
#CRAWLING
bearer_token = os.getenv('OD_BEARER_TOKEN')

api_key = os.getenv('OD_API_KEY') 
api_secret = os.getenv('OD_API_KEY_SECRET') 

access_token = os.getenv('OD_ACCESS_TOKEN')
access_secret = os.getenv('OD_ACCESS_TOKEN_SECRET')

auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
client = tweepy.Client(bearer_token=bearer_token, 
                       consumer_key=api_key, 
                       consumer_secret=api_secret, 
                       access_token=access_token, 
                       access_token_secret=access_secret,
                       wait_on_rate_limit=True)

user_id = 351535962
user_name = '@officialJKT48'

#PREPROC
stopwords = set(stopwords.words('indonesian'))
singkatan = pd.read_csv('kamus_singkatan.csv',sep=';',names=['kata','arti'])
df2 = {'kata': 'grgr', 'arti': 'gara gara'}
singkatan = singkatan.append(df2, ignore_index = True)
sing = singkatan.to_numpy()

def clean_tweet(tweet):
    if type(tweet) == float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    for t in range(len(temp)):
        if temp[t] in sing:
            idx=np.where(sing==temp[t])[0][0]
            temp[t] = sing[idx][1]
    temp = [w for w in temp if not w in stopwords and w != 'rt']
    temp = " ".join(word for word in temp[:512])
    temp = temp.strip()
    return temp

#EMOTION MODEL
path = "akahana/indonesia-emotion-roberta"
emotion = pipeline('text-classification', 
                     model=path)

#SENTIMENT MODEL
pretrained_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
sentiment = pipeline(
    "sentiment-analysis",
    model=pretrained_name,
    tokenizer=pretrained_name
)

def crawlQRTREP(tweet_id):

    tweet_list = []
    pagination_token = None

    #try:
    i=1
    while 1:
        print("Pagination :",i)
        i+=1
        p = client.get_quote_tweets(tweet_id, max_results=100,tweet_fields="public_metrics", pagination_token=pagination_token)
        try:
            print("Get",len(p.data),"data")
            print("Scrapped:",len(tweet_list))
            tweet_list.extend(p.data)
            if len(p.data) == 0 :
                break
        except:
            print("No data scrapped.")
            break
        if 'next_token' in p.meta:
            pagination_token = p.meta['next_token']
        else:
            break

#except:
    print("DONE CRAWLING QRT")

    pagination_token = None

#try:
    i=1
    while 1:
        print("Pagination :",i)
        i+=1
        query = f"conversation_id:{tweet_id} is:reply"
        p = client.search_recent_tweets(query= query,max_results=100,tweet_fields="public_metrics", next_token=pagination_token)
        print("Scrapped:",len(tweet_list))
        try:
            print("Get",len(p.data),"data")
            tweet_list.extend(p.data)
            if len(p.data) == 0 :
                break
        except:
            print("No data scrapped.")
            break
        if 'next_token' in p.meta:
            pagination_token = p.meta['next_token']
        else:
            break
#except:
    print("DONE CRAWLING REPLIES")

    return [i.data for i in tweet_list]
    
#PROCESSING
def processingTweet(list_tweet, hashtag):
    txt = """#JKT48SentimentAnalysis
#JKT48EmotionDetection

Analisis untuk hashtag : {hs} 

Data Gathered : {dg}

SENTIMENT 
Positive: {pos:.2f}%
Negative: {neg:.2f}%
Neutral: {neu:.2f}%

EMOTION
Takut: {tkt:.2f}%
Sedih: {sdh:.2f}%
Cinta: {cnt:.2f}%
Bahagia: {bhg:.2f}%
Marah: {mrh:.2f}%"""
    print("PROCESSING :",hashtag)
    print()
    list_t = [clean_tweet(str(i[1])) for i in list_tweet if clean_tweet(str(i[1])) !='' and i != '?']
    list_tt = list(dict.fromkeys(list_t))

    emotion_tt = []
    sentiment_tt = []
    start = int(0)
    batch_size = 1024
    end = batch_size
    batches = int(math.ceil(len(list_tt)/batch_size))
    for i in range(batches):
        print("START BATCH",i+1)
        tmp1 = emotion(list_tt[start:end])
        tmp2 = sentiment(list_tt[start:end])
        start = end
        end += batch_size
        emotion_tt.extend(tmp1)
        sentiment_tt.extend(tmp2)
        print(i+1,"DONE")

    #HASIL EMOTION DETECTION
    try:
        emot_tt = [i['label'] for i in emotion_tt]
        takut = emot_tt.count('TAKUT')
        sedih = emot_tt.count('SEDIH')
        cinta = emot_tt.count('CINTA')
        bahagia = emot_tt.count('BAHAGIA')
        marah = emot_tt.count('MARAH')
        print("EMOTION DETECTION :")
        print("TAKUT:",takut/len(emot_tt)*100)
        print("SEDIH:",sedih/len(emot_tt)*100)
        print("CINTA:",cinta/len(emot_tt)*100)
        print("BAHAGIA:",bahagia/len(emot_tt)*100)
        print("MARAH:",marah/len(emot_tt)*100)
        print("")

        #HASIL SENTIMENT
        sent_tt = [i['label'] for i in sentiment_tt]
        neg_t = sent_tt.count('negative')
        pos_t = sent_tt.count('positive')
        ntrl_t = sent_tt.count('neutral')
        print("Positive:",pos_t/len(sent_tt)*100)
        print("Negative:",neg_t/len(sent_tt)*100)
        print("Neutral:",ntrl_t/len(sent_tt)*100)


        #POST HASIL TWEET
        txt_twt = txt.format(
            pos=pos_t/len(sent_tt)*100,
            neg=neg_t/len(sent_tt)*100,
            neu=ntrl_t/len(sent_tt)*100,
            tkt=takut/len(emot_tt)*100,
            sdh=sedih/len(emot_tt)*100,
            cnt=cinta/len(emot_tt)*100,
            bhg=bahagia/len(emot_tt)*100,
            mrh=marah/len(emot_tt)*100,
            hs=hashtag,
            dg=len(list_tt)
        )
        
        hasil = []
        for i in range(len(list_tt)):
            hasil.append([list_tt[i],emot_tt[i],sent_tt[i]])

        df = pd.DataFrame(hasil,columns =['tweet','emotion','sentiment'])
        df.to_csv(hashtag+'.csv')
        
        print(txt_twt)
        tt = client.create_tweet(text=txt_twt)
        print("ID : ",tt.data['id'])
        print("Hashtag : ",hashtag)
        print("")
        
    except:
        pass
        

if __name__ == "__main__":
    hashtag = input("Input hashtag:")
    proceed_tweets = []
    last_tweet = None

    query = hashtag
    i=1
    pagination_token = None
    while 1:
        print("Pagination :",i)
        i+=1
        search_result = client.search_recent_tweets(query= query,max_results=100, next_token=pagination_token,tweet_fields="public_metrics")
        new_tweets = [(i.id,i.text,i.public_metrics) for i in search_result.data ]
        print("Scrapped:",len(new_tweets))
        try:
            print("Get",len(search_result.data),"data")
            proceed_tweets.extend(new_tweets)
            if len(search_result.data) == 0 :
                break
        except:
            print("No data scrapped.")
            break
        if 'next_token' in search_result.meta:
            pagination_token = search_result.meta['next_token']
        else:
            break

    tweets = []
    while len(proceed_tweets) > 0 :
        the_tweet = proceed_tweets.pop()
        if the_tweet[2]['reply_count'] > 0 or the_tweet[2]['quote_count'] > 0 :
            list_tweet = [(i['id'],i['text']) for i in crawlQRTREP(the_tweet[0])]
            list_tweet.append((the_tweet[0],the_tweet[1]))
            tweets.extend(list_tweet)
        #processingTweet(list_tweet, twt_id[0])
    processingTweet(tweets, hashtag)
    


