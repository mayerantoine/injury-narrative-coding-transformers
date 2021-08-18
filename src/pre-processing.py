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
import logging
import os
from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from pickle import load, dump


def _parse_args():
    parser = argparse.ArgumentParser()
       
    ## Hyperparams
    parser.add_argument("--data_path",type=str)
    parser.add_argument("--is_sample_dataset",default=False,action='store_true')
    parser.add_argument("--train_percentage",type=float,default=0.05)
    parser.add_argument("--max_len",type=int,default=45)
    parser.add_argument("--transformer_model",type=str,default='bert-base-uncased')
    


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
    print(is_sample)
    
    if is_sample:
        print(is_sample)
        train = get_samples(ratio = ratio, train=train)
    great_than2_classes = train['event'].value_counts()[train['event'].value_counts() >5].index 
    train_filter = train[train['event'].isin(great_than2_classes.to_list())]
    
    logging.info(f"nb classes in final data:{train_filter['event'].nunique()}")

    X = train_filter['text']
    y = train_filter['event']

    logging.info(f" X {X.shape} , y : {y.shape}")

    X_train_valid,X_test,y_train_valid, y_test = train_test_split(X,y,train_size=0.9,random_state=42,stratify=y)
    
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

def _tokenize_data(X_train_processed,X_valid_processed,transformer_model,MAX_LEN):
    
    if transformer_model == 'distilbert-base-uncased':
        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


    x_train = X_train_processed.to_list()
    x_valid = X_valid_processed.to_list()


    train_encodings = tokenizer(x_train, max_length=MAX_LEN, truncation=True, padding='max_length',return_tensors='tf')
    valid_encodings = tokenizer(x_valid, max_length=MAX_LEN, truncation=True, padding='max_length',return_tensors='tf')
    
    return train_encodings,valid_encodings


def _encoding_labels(y_train,y_valid):
    
    logging.info("Encoding Labels .....")
    encoder = LabelEncoder()
    encoder.fit(y_train)
    
    y_train_encode = np.asarray(encoder.transform(y_train))
    y_valid_encode = np.asarray(encoder.transform(y_valid))
    
    #data_path='/opt/ml/processing/output/processed/train/'
    data_path = './data/train/'
    os.makedirs(data_path,exist_ok=True)
    
    #test_data_path='/opt/ml/processing/output/processed/test/'
    test_data_path='./data/test/'
    os.makedirs(test_data_path,exist_ok=True)

    # save encoder for training
    dump(encoder,open(os.path.join(data_path,'encode.pkl'),'wb'))
    
    # save encoder for testing
    dump(encoder,open(os.path.join(test_data_path,'encode.pkl'),'wb'))
    
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
    

def _load_test_data(test_data_path):
    test_data = pd.read_csv(f'{test_data_path}/test_data.csv')
    test_data = test_data[~test_data['event'].isin([10,29,30,59,74])]
    test_data = test_data[['text','event']]
                          
    return test_data


def main():
    
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                        format='%(levelname)s: %(asctime)s %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.info("Start.....")
    logging.info("Parsing arguments")
    args, unknown = _parse_args()
    
    
    logging.info("Getting and splitting data")
    data_file = f"{args.data_path}/train.csv"
                          
    #data = get_data(data_file,is_sample=args.is_sample_dataset,ratio=args.train_percentage)
    data = get_data(data_file,is_sample=bool(args.is_sample_dataset) ,ratio=args.train_percentage)
    test_data = _load_test_data(args.data_path)
        
    X_train, y_train = data['train']
    X_valid, y_valid = data['valid']
    
    CLASSES = y_train.unique().tolist()
    
    print(CLASSES)
    
    logging.info("Preprocessing...")
    X_train_processed = preprocess_data(X_train)
    X_valid_processed = preprocess_data(X_valid)
    
    logging.info("Tokenization and encoding...")
    
    train_encodings, valid_encodings = _tokenize_data(X_train_processed,X_valid_processed,args.transformer_model,args.max_len)       
    y_train_encode, y_valid_encode = _encoding_labels(y_train,y_valid)
    
    logging.info("Create TF Dataset....")
    
    train_tfdataset = construct_tfdataset(train_encodings, y_train_encode)
    valid_tfdataset = construct_tfdataset(valid_encodings, y_valid_encode)
    
    logging.info("Saving train and valid TF Records...")
    
    # training data
    #_save_feature_as_tfrecord(train_tfdataset,'/opt/ml/processing/output/processed/train/train.tfrecord')
    _save_feature_as_tfrecord(train_tfdataset,'./data/train/train.tfrecord')

                          
    #validation data
    #_save_feature_as_tfrecord(valid_tfdataset,'/opt/ml/processing/output/processed/validation/valid.tfrecord')
    _save_feature_as_tfrecord(valid_tfdataset,'./data/valid/valid.tfrecord')
 
    logging.info("Saving test dataset...")                      
    X_test, y_test = test_data['text'], test_data['event']
    X_test_processed = preprocess_data(X_test)
                          
    text_processed = pd.DataFrame({'text':X_test_processed,'event':y_test})
    
    #test data
    #text_processed.to_csv('./opt/ml/processing/output/processed/test/text_processed.csv')
    text_processed.to_csv('./data/test/text_processed.csv')
    
    logging.info("Complete")



if __name__ == "__main__":
    main()