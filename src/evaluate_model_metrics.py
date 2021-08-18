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
    parser.add_argument("--input_data", type=str, default="/opt/ml/processing/input/data")
    parser.add_argument("--input_model",type=str,default="/opt/ml/processing/input/model")
    parser.add_argument("--output_data",type=str,default="/opt/ml/processing/output")
    parser.add_argument("--max_len", type=str, default=45)

    
    return  parser.parse_known_args()


def compute_metrics(pred):
    labels = pred.true_event
    preds = pred.preds_event
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    precision = precision_score(labels,preds,average='macro')
    recall = recall_score(labels,preds,average='macro')
    f1 = f1_score(labels,preds,average='macro')
    return {
        'accuracy': acc,
        'balanced_accuracy':bal_acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }





def construct_tfdataset(encodings, y=None):
    if y is not None:
        return tf.data.Dataset.from_tensor_slices((dict(encodings),y))
    else:
        # this case is used when making predictions on unseen samples after training
        return tf.data.Dataset.from_tensor_slices(dict(encodings))

    
def _load_encoder(test_dir):
    # load the encoder
    encoder_file = os.path.join(test_dir,'encode.pkl') 
    encoder = load(open(encoder_file, 'rb'))
    
    return encoder


def extract_data_load_model(input_model,model_tar_path,model_name):
    model_class = 'BaseBERT'
    t = tarfile.open(model_tar_path, 'r:gz')
    t.extractall(path=f"{input_model}")
    _model = tf.keras.models.load_model(f"{input_model}/{model_class}",custom_objects={'AdamWeightDecay':AdamWeightDecay})
    
    return _model



def _batch_predict(model, encoder,model_name, max_len,x,y) :
    tkzr = AutoTokenizer.from_pretrained(model_name)
    encodings_x =  tkzr(x, max_length=max_len, truncation=True, padding='max_length',return_tensors='tf')
    tfdataset = construct_tfdataset(encodings_x).batch(32)
    preds = model.predict(tfdataset)
    predictions_encode = pd.DataFrame(data=preds).apply(lambda x: np.argmax(x),axis=1)
    categories = encoder.classes_.tolist()
    predictions_event= predictions_encode.apply(lambda x:categories[x])


    print(len(predictions_event))
    results = pd.DataFrame({'text': x,
                      'true_event':y,
                      'preds_encode': predictions_encode,
                      'preds_event':predictions_event
                            })


    return results



def _create_predictor(model, encoder,model_name, max_len,text):
    tkzr = AutoTokenizer.from_pretrained(model_name)
    x = [text]
    encodings =  tkzr(x, max_length=max_len, truncation=True, padding='max_length',return_tensors='tf')
    tfdataset = construct_tfdataset(encodings)
    tfdataset = tfdataset.batch(1)
    preds = model.predict(tfdataset)
    categories = encoder.classes_.tolist()
    enc = np.argmax(preds[0])
    
    return {     'text' : x,
                 'predict_proba' : preds[0][np.argmax(preds[0])],
                 'predicted_class' : categories[np.argmax(preds)]                             
           }


def main():
    
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                        format='%(levelname)s: %(asctime)s %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.info("Start.....")
    logging.info("Parsing arguments")
    
    args, unknown = _parse_args()
    
    logging.info("Create sagemaker session")    
    sagemaker_session = sagemaker.Session()
    
    logging.info("input_data: {}".format(args.input_data))
    logging.info("input_model: {}".format(args.input_model))

    logging.info("Listing contents of input model dir: {}".format(args.input_model))
    input_files = os.listdir(args.input_model)
    for file in input_files:
        print(file)
    
    logging.info("Loading model..")
    model_tar_path = "{}/model.tar.gz".format(args.input_model)
    _model = extract_data_load_model(args.input_model,model_tar_path,'bert-base-uncased')
    
    logging.info("Listing contents of input data dir: {}".format(args.input_data))
    input_files = os.listdir(args.input_data)
    
    
    logging.info("Loading encoder..")
    encode = _load_encoder(args.input_data)
   # logging.info("Encoder class:",encode.classes_.tolist())
    
    logging.info(" Load and preprocess Test data")
    test_data_path = "{}/test_processed.csv".format(args.input_data)
    
    test_processed = pd.read_csv(test_data_path)
    
    x_test = test_processed['text'].tolist()
    y_test = test_processed['event'].tolist()
    
    logging.info("Batch predict test results")

    
    text = x_test[0]
    pred = _create_predictor(_model,encode,'bert-base-uncased',45,text)
    print(pred)
          
    results_test = _batch_predict(_model,encode,'bert-base-uncased',45,x_test,y_test)
    print(results_test)
        
    

if __name__ == "__main__":
    main()