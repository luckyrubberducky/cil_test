# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:56:49 2020

@author: zeno
"""

import re
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification, AdamW, pipeline
import torch

with open("train_pos.txt") as f:
    pos_tweets = f.readlines()

with open("train_neg.txt") as f:
    neg_tweets = f.readlines()
pos_tweet = pos_tweets[0]
neg_tweet = neg_tweets[0]
tweets = []

for tweet in pos_tweets[0:10000]:
    #tweets.append(re.findall(',([^"]*),', tweet)[0])
    tweets.append(tweet)
    
tweet = tweets[0]
    
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')

optimizer = AdamW(model.parameters(), lr=1e-5)


classes = ["positive", "negative"]

input = tokenizer(pos_tweet, return_tensors="pt")

print(input)

outputs = model(**input)

print(outputs[0])