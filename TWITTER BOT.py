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
bearer_token = os.getenv('BEARER_TOKEN')

api_key = os.getenv('API_KEY') 
api_secret = os.getenv('API_KEY_SECRET') 

access_token = os.getenv('ACCESS_TOKEN')
access_secret = os.getenv('ACCESS_TOKEN_SECRET')

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

#CRAWLING QRT & REP
def crawlQRTREP(tweet_id):

    tweet_list = []
    pagination_token = None

    #try:
    i=1
    while 1:
        print("Pagination :",i)
        i+=1
        p = client.get_quote_tweets(tweet_id, max_results=100, pagination_token=pagination_token)
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
        p = client.search_recent_tweets(query= query,max_results=100, next_token=pagination_token)
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
def processingTweet(list_tweet, tweet_id):
    txt = """#JKT48SentimentAnalysis
#JKT48EmotionDetection
Analisis Reply & Qoute Retweet dari tweet: 
https://twitter.com/officialJKT48/status/{tweet_id}

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
    txt2 = """#JKT48SentimentAnalysis
#JKT48EmotionDetection
Analisis Reply & Qoute Retweet dari tweet: 
https://twitter.com/officialJKT48/status/{tweet_id}

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
    print("PROCESSING :",tweet_id)
    print()
    list_t = [clean_tweet(str(i['text'])) for i in list_tweet if clean_tweet(str(i['text'])) !='' and i != '?']
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
        try:
            txt_twt = txt2.format(
                pos=pos_t/len(sent_tt)*100,
                neg=neg_t/len(sent_tt)*100,
                neu=ntrl_t/len(sent_tt)*100,
                tkt=takut/len(emot_tt)*100,
                sdh=sedih/len(emot_tt)*100,
                cnt=cinta/len(emot_tt)*100,
                bhg=bahagia/len(emot_tt)*100,
                mrh=marah/len(emot_tt)*100,
                tweet_id=tweet_id,
                dg=len(list_tt)
            )
            
            print(txt_twt)
            tt = client.create_tweet(text=txt_twt)
        except:
            txt_twt = txt.format(
                pos=pos_t/len(sent_tt)*100,
                neg=neg_t/len(sent_tt)*100,
                neu=ntrl_t/len(sent_tt)*100,
                tkt=takut/len(emot_tt)*100,
                sdh=sedih/len(emot_tt)*100,
                cnt=cinta/len(emot_tt)*100,
                bhg=bahagia/len(emot_tt)*100,
                mrh=marah/len(emot_tt)*100,
                tweet_id=tweet_id
            )
            print(txt_twt)
            tt = client.create_tweet(text=txt_twt)
        print("Twitter ID : ",tt.data['id'])
        print("")
        
        hasil = []
        for i in range(len(list_tt)):
            hasil.append([list_tt[i],emot_tt[i],sent_tt[i]])

        df = pd.DataFrame(hasil,columns =['tweet','emotion','sentiment'])
        df.to_csv(tt.data['id']+'.csv')
    except:
        pass


#THREAD DEF
def processBOT(tweet_ids):
    ids = tweet_ids
    while len(ids) > 0:
        twt_id = ids.pop()
        schedule = twt_id[1] + timedelta(hours=6)
        delta = schedule - datetime.now(timezone.utc)
        if int(delta.total_seconds() > 0):
            print(twt_id[0],"will be processed",datetime.now()+delta)
            time.sleep(delta.total_seconds())
        
        list_tweet = crawlQRTREP(twt_id[0])
        processingTweet(list_tweet, twt_id[0])
        with open('tweets.txt', "a", encoding="utf-8") as f:
            f.write(str(twt_id[0])+"\n")
        

if __name__ == "__main__":
    proceed_tweets = []
    last_tweet = None
    while 1:
        #our_tweet = client.get_users_tweets(1326009943602814977, max_results=5,exclude=['replies','retweets'],tweet_fields='created_at')
        #our_id = [(i.id,i.created_at) for i in our_tweet.data]
        #proceed_tweets = list(set(proceed_tweets) - set(our_id))

        user_tweets = client.get_users_tweets(351535962, max_results=10,exclude=['replies','retweets'],tweet_fields='created_at')
        tweet_user_ids = [(i.id,i.created_at) for i in user_tweets.data]
        current_time = datetime.now()
        print("==========================================")
        print("")
        print("Current Time:", current_time)
        with open('tweets.txt', "r", encoding="utf-8") as f:
            data_t = f.read()
            existed_tweet = [int(i) for i in data_t.split("\n") if i != '']

        new_tweets = [i[0] for i in tweet_user_ids]
        p_tweets = [i[0] for i in proceed_tweets]
        tweet_ids = list(set(new_tweets) - set(existed_tweet) - set(p_tweets))
        if len(tweet_ids) > 0 :
            tweets = tweet_user_ids[:len(tweet_ids)]
            proceed_tweets = tweets + proceed_tweets
            print("Found new tweet!")
            print("Tweet IDs:",tweet_ids)
            print("Will be processed in 6 hours prior to the tweet.")
            thrd = threading.Thread(target=processBOT, args=(tweets,))
            thrd.start()
        else:
            print("No new tweet from @officialJKT48")
        time.sleep(2)
        print("")
        #print("On queue:",p_tweets)
        try:
            for i in range(len(proceed_tweets),1):
                schedule = proceed_tweets[-i][1] + timedelta(hours=6)
                delta = schedule - datetime.now(timezone.utc)
                if int(delta.total_seconds() > 0):
                    last_tweet = proceed_tweets[-i]
                    break
            if last_tweet != None:
                schedule = last_tweet[1] + timedelta(hours=6)
                delta = schedule - datetime.now(timezone.utc)
                print("Next Processing time:",datetime.now()+delta,"for",last_tweet[0])
        except:
            pass
        print("15 Minutes sleep")
        print("Next fetching time:",datetime.now() + timedelta(minutes=15))
        time.sleep(900)


