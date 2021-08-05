import pandas as pd
import tensorflow as tf
import re
import nltk
import string
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras import activations, optimizers, losses
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import numpy as np
import sagemaker
from sagemaker import get_execution_role
import joblib 
import collections
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
       
    ## Hyperparams
    parser.add_argument("--data_path",type=str)
    parser.add_argument("--is_sample",type=int)
    parser.add_argument("--ratio",type=str)
    parser.add_argument("--is_test_split",type=int,default=45)

    return  parser.parse_known_args()



def get_samples(ratio,train):
    great_than2_classes = train['event'].value_counts()[train['event'].value_counts() >2].index
    train = train[train['event'].isin(great_than2_classes.to_list())]
    train_samples, _ = train_test_split(train,train_size=ratio,random_state=42,stratify=train['event'])
    print("nb classes in samples",train_samples['event'].nunique())
    print("nb oservations:",train_samples.shape)
    #print(train_samples['event'].value_counts())

    return train_samples


def get_data(is_sample=None,ratio=None,is_test_split=False):
  
    train = pd.read_csv('train.csv')
    if is_sample:
        train = get_samples(ratio = ratio, train=train)
    great_than2_classes = train['event'].value_counts()[train['event'].value_counts() >5].index 
    train_filter = train[train['event'].isin(great_than2_classes.to_list())]
    
    print("nb classes in final data:",train_filter['event'].nunique())

    X = train_filter['text']
    y = train_filter['event']

    print(f"X.shape {X.shape} y.shape : {y.shape}")

    X_train_valid,X_test,y_train_valid, y_test = train_test_split(X,y,train_size=0.8,random_state=42,stratify=y)
    
    if is_test_split:
        X_train,X_valid,y_train,y_valid = train_test_split(X_train_valid,y_train_valid,train_size=0.8,random_state=42,stratify=y_train_valid)

    if is_test_split :
        print(f"X_train shape {X_train.shape} y_train shape : {y_train.shape}")
        print(f"X_valid shape {X_valid.shape} y_valid shape : {y_valid.shape}")
        print(f"X_test shape {X_test.shape} y_test shape : {y_test.shape}")
        
        return {
          'train': (X_train,y_train),
          'valid': (X_valid,y_valid),
          'test': (X_test,y_test)
      }

    else:
        print(f"X_train shape {X_train_valid.shape} y_train shape : {y_train_valid.shape}")
        print(f"X_valid shape {X_test.shape} y_valid shape : {y_test.shape}")
        
        return {
          'train': (X_train_valid,y_train_valid),
          'valid': (X_test,y_test),
      }

    
def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    # convert text to lowercase
    text = text.strip().lower()

    # remove all non-ASCII characters:
    text = re.sub(r'[^\x00-\x7f]', r'', text)

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    text = " ".join(text.split())
    return text

def remove_useless_words(text,useless_words):
    sentence = [word for word in word_tokenize(text)]
    sentence_stop = [word for word in sentence if word not in useless_words]

    text = " ".join(sentence_stop)

    return text


def preprocess_data(X):
    """ Preprocess : cleaning and remove stop words"""

    X = X.apply(lambda x: re.sub(r'\d+', '', x))
    X = X.apply(lambda x: clean_text(x))

    stopwords = nltk.corpus.stopwords.words('english')
    useless_words = stopwords + list(string.punctuation) + ['yom', 'yof', 'yowm', 'yf', 'ym', 'yo']
    # print("useless word : ",useless_words)
    X = X.apply(lambda x: remove_useless_words(x,useless_words))

    return X



def main():
    args, unknown = _parse_args()
    print("input data: ",args.data_path)
    
    data = get_data(is_sample=True,ratio=0.05)
    
    #data = get_data(is_test_split=True)
    
    X_train, y_train = data['train']
    X_valid, y_valid = data['valid']
    
    CLASSES = y_train.unique().tolist()
    
    print(CLASSES)
    
    X_train_processed = preprocess_data(X_train)
    X_valid_processed = preprocess_data(X_valid)
    
    #save 
    #[train(X_train_processed,y_train)
    #[valid(X_valid_processed,y_valid)


if __name__ == "__main__":
    main()