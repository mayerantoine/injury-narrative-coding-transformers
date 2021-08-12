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


save_local_data_path = './data'


def _parse_args():
    parser = argparse.ArgumentParser()
    ## Experiments parameters
    parser.add_argument("--data_path", type=str, default='./train.csv')
    parser.add_argument("--is_sample", type=str, default='True')
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--bucket", type=str, default='cdc-cdh-sagemaker-s3fs-dev')
    parser.add_argument("--transformer_model",type=str,default='distilbert-base-uncased')
    parser.add_argument("--wait",type=str,default='False')
    
    
    ## Hyperparams
    parser.add_argument("--model_name",type=str,default='DistilBERT')
    parser.add_argument("--num_records",type=int)
    parser.add_argument("--optimizer",type=str,default='adam')
    parser.add_argument("--max_len",type=int,default=45)
    parser.add_argument("--learning_rate",type=float,default=5e-5)
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--valid_batch_size",type=int,default=8)
    parser.add_argument("--steps_per_epoch",type=int)
    parser.add_argument("--validation_steps",type=int)
    
    return  parser.parse_known_args()


def get_samples(ratio,train):
    great_than2_classes = train['event'].value_counts()[train['event'].value_counts() >2].index
    train = train[train['event'].isin(great_than2_classes.to_list())]
    train_samples, _ = train_test_split(train,train_size=ratio,random_state=42,stratify=train['event'])
    #print("nb classes in samples",train_samples['event'].nunique())
    #print("nb oservations:",train_samples.shape)
    #print(train_samples['event'].value_counts())

    return train_samples


def get_data(data_file,is_sample=None,ratio=None,is_test_split=False):
  
    train = pd.read_csv(data_file)
    if is_sample:
        train = get_samples(ratio = ratio, train=train)
    great_than2_classes = train['event'].value_counts()[train['event'].value_counts() >5].index 
    train_filter = train[train['event'].isin(great_than2_classes.to_list())]
    
    logging.info(f"nb classes in final data:{train_filter['event'].nunique()}")

    X = train_filter['text']
    y = train_filter['event']

    logging.info(f" X {X.shape} , y : {y.shape}")

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
        logging.info(f"X_train shape {X_train_valid.shape} y_train shape : {y_train_valid.shape}")
        logging.info(f"X_valid shape {X_test.shape} y_valid shape : {y_test.shape}")
        
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


def _load_data():
    pass

def _tokenize_data(x_train,x_valid,transformer_model,max_len):
    
    #if transformer_model == 'distilbert-base-uncased':
    #    tokenizer = DistilBertTokenizerFast.from_pretrained(transformer_model)
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)

    x_train = x_train.to_list()
    x_valid = x_valid.to_list()


    train_encodings = tokenizer(x_train, max_length=max_len, truncation=True, padding='max_length',return_tensors='tf')
    valid_encodings = tokenizer(x_valid, max_length=max_len, truncation=True, padding='max_length',return_tensors='tf')
    
    return train_encodings,valid_encodings


def _encoding_labels(y_train,y_valid):
    
    #print("Encoding Labels .....")
    encoder = LabelEncoder()
    encoder.fit(y_train)
    
    y_train_encode = np.asarray(encoder.transform(y_train))
    y_valid_encode = np.asarray(encoder.transform(y_valid))
    
    data_path='./data'
    os.makedirs(data_path,exist_ok=True)


    dump(encoder,open(os.path.join(data_path,'encode.pkl'),'wb'))
    
    return y_train_encode, y_valid_encode
    
    
def construct_tfdataset(encodings, y=None):
    if y is not None:
        return tf.data.Dataset.from_tensor_slices((dict(encodings),y))
    else:
        # this case is used when making predictions on unseen samples after training
        return tf.data.Dataset.from_tensor_slices(dict(encodings))
    

def _save_feature_as_tfrecord(tfdataset,file_path):
    """ Helper function to save the tf dataset as tfrecords"""
    

    def single_example_data(encoding,label_id):

        input_ids =encoding['input_ids']
        attention_mask = encoding['attention_mask']

        tfrecord_features = collections.OrderedDict()

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        tfrecord_features['input_ids'] = _int64_feature(input_ids)
        tfrecord_features['attention_mask'] = _int64_feature(attention_mask)
        tfrecord_features['label_ids'] =  _int64_feature([label_id])

        _example = tf.train.Example(features=tf.train.Features(feature=tfrecord_features))

        return _example.SerializeToString()

    def data_generator():
        for features in tfdataset:
            yield single_example_data(*features)

    serialized_tfdataset = tf.data.Dataset.from_generator(
        data_generator, output_types=tf.string, output_shapes=())        

    writer = tf.data.experimental.TFRecordWriter(file_path)
    writer.write(serialized_tfdataset)
    


def main():
    
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                        format='%(levelname)s: %(asctime)s %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.info("Start.....")
    logging.info("Parsing arguments")
    args, unknown = _parse_args()
    
      
    logging.info("Create sagemaker session")    
    sagemaker_session = sagemaker.Session(default_bucket=args.bucket)
    role = sagemaker.get_execution_role()
    default_bucket = sagemaker_session.default_bucket()
    
    logging.info("Getting and splitting data")
    data = get_data(args.data_path,is_sample=args.is_sample,ratio=args.ratio)
        
    X_train, y_train = data['train']
    X_valid, y_valid = data['valid']
    
        
    CLASSES = y_train.unique().tolist()
    
    logging.info(f"Number of classes: {CLASSES}")
    logging.info("Preprocessing training data...")
    X_train_processed = preprocess_data(X_train)
    
    logging.info("Preprocessing validation data...")
    X_valid_processed = preprocess_data(X_valid)
    
    print(X_train_processed.head())
    
    logging.info("Tokenize and encode data...")
    train_encodings, valid_encodings = _tokenize_data(X_train_processed,X_valid_processed,args.transformer_model,args.max_len)       
    y_train_encode, y_valid_encode = _encoding_labels(y_train,y_valid)
    
    logging.info("Create tfdataset...")
    train_tfdataset = construct_tfdataset(train_encodings, y_train_encode)
    valid_tfdataset = construct_tfdataset(valid_encodings, y_valid_encode)
    
    # training data
    logging.info("Saving training tfrecords....")
    _save_feature_as_tfrecord(train_tfdataset,'./data/train.tfrecord')

    #validation data
    logging.info("Saving validation tfrecords....")
    _save_feature_as_tfrecord(valid_tfdataset,'./data/valid.tfrecord')
    
    
    logging.info("Upload data to S3")
    
    data_prefix = 'projects/project006/injury-data'
    tfrecord_data_location = sagemaker_session.upload_data(path = './data',
                                                      bucket = args.bucket,
                                                      key_prefix = data_prefix)
    print(tfrecord_data_location)
    
    output_prefix = 'projects/project006/output'
    output_path = 's3://{}/{}/'.format(args.bucket, output_prefix )

    
    num_records = len(X_train)
    num_valid_records = len(X_valid)
    max_len = args.max_len
    epochs =args.epochs
    batch_size =args.batch_size
    valid_batch_size = args.batch_size
    steps_per_epoch = num_records // batch_size
    validation_steps = num_valid_records // valid_batch_size
    learning_rate = args.learning_rate
    optimizer = 'adam'
    
    job_name_prefix = f"training-{args.model_name,}"
    timestamp = strftime("-%m-%d-%M-%S", gmtime())

    job_name = job_name_prefix + timestamp

    _estimator = HuggingFace(
            base_job_name  = job_name,
            entry_point="train.py",
            source_dir = "./src/",
            role=role,
            instance_count=1,
            volume_size = 5,
            max_run = 18000,
            instance_type='ml.p3.2xlarge',
            transformers_version = "4.4",
            tensorflow_version  = "2.4",
            py_version="py37",
            output_path = output_path,
            hyperparameters = {
                    "model_name": args.model_name,
                    "num_records":  num_records,
                    "max_len":max_len,
                    "epochs":int(epochs),
                    "learning_rate":float(learning_rate),
                    "batch_size":int(batch_size),
                    "valid_batch_size":valid_batch_size,
                    "steps_per_epoch": steps_per_epoch,
                    "validation_steps": validation_steps,
                    "optimizer":optimizer
                    },
            metric_definitions = [{'Name':'train:loss','Regex':'loss: ([0-9\\.]+)'},
                                        {'Name':'train:accuracy','Regex':'acc: ([0-9\\.]+)'},
                                        {'Name':'validation:loss','Regex':'val_loss: ([0-9\\.]+)'},
                                        {'Name':'validation:accuracy','Regex':'val_acc: ([0-9\\.]+)'}],
            enable_sagemaker_metrics = True
        )
    
    logging.info('Start training...')
    #_estimator.fit({'train':tfrecord_data_location}, wait=False)
        

if __name__ == "__main__":
    main()