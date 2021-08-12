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



def _load_data(train_dir,MAX_LEN,epochs,batch_size,valid_batch_size,steps_per_epoch,validation_steps):
    """ Helper function to load,parse and create input data pipeline from TFRecords"""
          
    train_file = os.path.join(train_dir,"train.tfrecord") 
    valid_file = os.path.join(train_dir,"valid.tfrecord")
    
    # Create a description of the features.
    feature_description = {
        'input_ids': tf.io.FixedLenFeature([MAX_LEN], tf.int64),
        'attention_mask': tf.io.FixedLenFeature([MAX_LEN], tf.int64),
        'label_ids': tf.io.FixedLenFeature([], tf.int64),
    }
        
    def _parse_function(example_proto):

        # Parse the input `tf.train.Example` proto using the dictionary above.
        parsed  = tf.io.parse_single_example(example_proto, feature_description)

        return {'input_ids':parsed['input_ids'],'attention_mask':parsed['attention_mask']},parsed['label_ids']
        
    
    train_dataset = tf.data.TFRecordDataset(train_file)
    train_dataset = train_dataset.repeat(epochs * steps_per_epoch)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(_parse_function,num_parallel_calls=tf.data.AUTOTUNE)
    
    train_dataset = train_dataset.batch(batch_size)
    
    
    
    valid_dataset = tf.data.TFRecordDataset(valid_file)
    valid_dataset = valid_dataset.repeat(epochs * validation_steps)
    valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.map(_parse_function,num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(valid_batch_size)
    
   
    return train_dataset, valid_dataset

def download_model(job_name,sagemaker_session):
    
    modeldesc = sagemaker_session.describe_training_job(job_name)
    s3_model_path = modeldesc['ModelArtifacts']['S3ModelArtifacts']

    
    os.makedirs(f"./output/model/{job_name}/",exist_ok=True)

    S3Downloader.download(
        s3_uri=s3_model_path, # s3 uri where the trained model is located
        local_path=f"./output/model/{job_name}/", # local path where *.targ.gz is saved
        sagemaker_session=sagemaker_session # sagemaker session used for training the model
    )
    
    return s3_model_path, modeldesc['HyperParameters']


def extract_data_load_model(job_name,model_name):

    t = tarfile.open(f'./output/model/{job_name}/model.tar.gz', 'r:gz')
    t.extractall(path=f'./output/model/{job_name}')
    _model = tf.keras.models.load_model(f"./output/model/{job_name}/{model_name}",custom_objects={'AdamWeightDecay':AdamWeightDecay})
    
    return _model
    

def _evaluate_model(loaded_model,train_dataset,valid_dataset,hp):
    print("Evaluating Training data...")
    train_score = loaded_model.evaluate(train_dataset,
                                     steps =  int(hp['steps_per_epoch']),
                                     batch_size=int(hp['batch_size']))

    print("Training loss: ", train_score[0])
    print("Training accuracy: ", train_score[1])

    print("Evaluating Validation data...")
    valid_score = loaded_model.evaluate(valid_dataset, steps =  int(hp['validation_steps']),batch_size= int(hp['valid_batch_size']))
    print("Validation loss: ", valid_score[0])
    print("Validation accuracy: ", valid_score[1])
    
    
def _batch_predict(model, encoder,model_name, max_len,x,y) :
    tkzr = AutoTokenizer.from_pretrained(model_name)
    encodings_x =  tkzr(x, max_length=MAX_LEN, truncation=True, padding='max_length',return_tensors='tf')
    tfdataset = construct_tfdataset(encodings_x).batch(32)
    preds = loaded_model.predict(tfdataset)
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