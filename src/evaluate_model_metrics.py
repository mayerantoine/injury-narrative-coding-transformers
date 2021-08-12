import sys
import subprocess
    
# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nltk'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])

import tensorflow as tf
import transformers
import nltk 
    
# Get Version information
print("Tensorflow: {0}".format(tf.__version__),'\n')
print("Hugging Face transfomers: {0}".format(transformers.__version__),'\n')
print("NLTK: {0}".format(nltk.__version__),'\n')

import os
import numpy as np
import pandas as pd
import argparse
import sagemaker
import joblib 
import collections
import re
import string
from nltk import word_tokenize
from pickle import load, dump
from sklearn.model_selection import train_test_split
from tensorflow.keras import activations, optimizers, losses
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sagemaker import get_execution_role
from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
from transformers import  TFDistilBertForSequenceClassification,DistilBertConfig
from transformers import TFBertForSequenceClassification, BertConfig
from transformers import TFRobertaForSequenceClassification, RobertaConfig
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from transformers.optimization_tf import WarmUp, AdamWeightDecay
from sklearn.preprocessing import LabelEncoder
import time
from time import gmtime, strftime
from sagemaker.huggingface import HuggingFace
import logging 
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,balanced_accuracy_score
from transformers.optimization_tf import AdamWeightDecay
import tarfile
from sagemaker.s3 import S3Downloader

def _parse_args():
    parser = argparse.ArgumentParser()
    ## Experiments parameters
    parser.add_argument("--job_name", type=str, default='')
    parser.add_argument("--train_dir", type=str, default='./data')

    
    return  parser.parse_known_args()



def main():
    
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                        format='%(levelname)s: %(asctime)s %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.info("Start.....")
    logging.info("Parsing arguments")
    
    args, unknown = _parse_args()
    
    logging.info("Create sagemaker session")    
    sagemaker_session = sagemaker.Session()
    
    logging.info("Download model from S3")  
    s3_model_artifact , hp = download_model(args.job_name,sagemaker_session)
    
    logging.info("Loading data..")  
    train_dataset,valid_dataset = _load_data(args.train_dir,int(hp['max_len']),
                                             int(hp['epochs']),
                                             int(hp['batch_size']),
                                             int(hp['valid_batch_size']),
                                             int(hp['steps_per_epoch']),
                                             int(hp['validation_steps']))
    
    logging.info("Extract and loading the model...") 
    model_name = hp['model_name'].strip('"')
    
    loaded_model = extract_data_load_model(args.job_name,model_name)
        
        
    _evaluate_model(loaded_model,train_dataset,valid_dataset,hp)
    

if __name__ == "__main__":
    main()