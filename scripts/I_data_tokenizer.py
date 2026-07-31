# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:19:07 2024

@author: Flavio Simoes da Silva

Purpose: Master's Thesis 
    Portfolio Optimization and NLP:
    Algorithmic trading with Black-Litterman and views from stock market sentiment coming from StockTwits.
    
    Supervisor: M.-A Divernois, Ph.D
"""

import pandas as pd
import numpy as np
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification


#=========================== Device Setup ===========================
print("CUDA Available:", torch.cuda.is_available()) # check for Nvidia graphics card compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)


#=========================== Load Data ===========================
# it takes a long time to load because there are 90mio tweets. You need RAM !!
# it would also be possible to load these two files jointly.

df = pd.read_pickle('df_withcorona_clean_1_with_proba_opti_and_hour.pkl') 
# df = pd.read_pickle('df_withcorona_clean_2_with_proba_opti_and_hour.pkl')

# can also be read together
'''
df = pd.concat([pd.read_pickle('df_withcorona_clean_1_with_proba_opti_and_hour.pkl'), 
                pd.read_pickle('df_withcorona_clean_2_with_proba_opti_and_hour.pkl')]) 
'''


#=========================== Data Manipulation ===========================
# Tickers used in the 'StockTwits Classified Sentiment and Stock Returns' paper
SCSSR_tickers = [
    "AAPL", "AMD", "AMRN", "AMZN", "BABA", "BAC", "BB", "FB", "GLD",
    "IWM", "JNUG", "MNKD", "NFLX", "PLUG", "QQQ", "SPY", "TSLA", "TWTR", "UVXY"
]

filtered_df = df[df["ticker"].isin(SCSSR_tickers)].copy()
filtered_df['day'] = pd.to_datetime(filtered_df['date']).dt.date
df_SCSSR = filtered_df[['day', 'ticker', 'sent', 'clean_text']] # take only the necessary columns for the model

print('df_SCSSR is done')







#=========================== Model & Tokenizer Setup ===========================
model_name = "StephanAkkerman/FinTwitBERT-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device) # to(device) ensure that the model runs on the GPU (if available)


#=========================== Batch Sentiment Analysis Function ===========================
def batch_fintwit_sentiment(texts, batch_size=64):
    """
    Processes tweets in batches to optimize speed.
    
    Parameters:
    ----------
        texts (list): list of tweets
        batch_size (int): number of tweets per batch
    
    Returns:
    ----------
        DataFrame with sentiment probabilities
    """    
    start_time = time.time()
    
    # filter out empty or NaN texts
    valid_texts = [text if isinstance(text, str) and text.strip() != "" else "" for text in texts]
    
    num_batches = len(valid_texts) // batch_size + (len(valid_texts) % batch_size != 0)
    all_scores = []

    for i in range(num_batches):
        batch_texts = valid_texts[i * batch_size : (i + 1) * batch_size]

        with torch.no_grad():
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
            
            end_time = time.time()  
            print(f"Cumulated time taken for prediction: {end_time - start_time:.4f} seconds")

            # collect the sentiment scores
            for prob in probabilities:
                scores = {
                    "neutral": prob[0],
                    "bullish": prob[1],
                    "bearish": prob[2],
                    "predicted_sentiment": ["neutral", "bullish", "bearish"][np.argmax(prob)]
                }
                all_scores.append(scores)
    
    end_time = time.time()
    print(f"Processed {len(texts)} tweets in {end_time - start_time:.2f} seconds")
    
    return pd.DataFrame(all_scores)


#===================== Apply Batch Processing to DataFrame ====================
# df_SCSSR = df_SCSSR.head(50000).copy()

start_time = time.time()
sentiment_results = batch_fintwit_sentiment(df_SCSSR["clean_text"].tolist(), batch_size=64)
end_time = time.time()

df_SCSSR = pd.concat([df_SCSSR.reset_index(drop=True), sentiment_results.reset_index(drop=True)], axis=1)

df_SCSSR["fintwit_score"] = df_SCSSR["bullish"] - df_SCSSR["bearish"]

print(f"Total processing time for {len(df_SCSSR)} tweets: {end_time - start_time:.2f} seconds")
print(f"Average time per tweet: {(end_time - start_time) / len(df_SCSSR):.4f} seconds")

print(df_SCSSR["predicted_sentiment"].value_counts())
print('Sentiment analysis done')


#====================== Download to pickle for later use ======================
# df_SCSSR.to_pickle("df_SCSSR_1_with_sentiment.pkl")
# df_SCSSR.to_csv("df_SCSSR_1_with_sentiment.csv")
# df_SCSSR.to_pickle("df_SCSSR_2_with_sentiment.pkl")
# df_SCSSR.to_csv("df_SCSSR_2_with_sentiment.csv")