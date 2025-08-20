import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def clean_tweet(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub("[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    return " ".join(tokens)

def load_and_preprocess_data(train_path) :#test_path):
    train = pd.read_csv(train_path)
    #test = pd.read_csv(test_path)

    train["clean_tweet"] = train["tweet"].apply(clean_tweet)
    #test["clean_tweet"] = test["tweet"].apply(clean_tweet)

    return train

def split_and_tokenize(train_df, tokenizer, max_length=60):
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["clean_tweet"], train_df["label"], test_size=0.3, random_state=42
    )

    train_enc = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='tf')
    val_enc = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='tf')

    return train_enc, val_enc, train_labels.tolist(), val_labels.tolist()
