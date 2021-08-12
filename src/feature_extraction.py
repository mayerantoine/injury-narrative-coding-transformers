import re
import nltk
import string
import numpy as np
import sagemaker
import pandas as pd
import tensorflow as tf
import joblib 
import collections
import argparse
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras import activations, optimizers, losses
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sagemaker import get_execution_role
from transformers import  DistilBertTokenizerFast,BertTokenizerFast,RobertaTokenizerFast
from sklearn.preprocessing import LabelEncoder
from pickle import dump

def _parse_args():
    parser = argparse.ArgumentParser()
       
    ## Hyperparams
    parser.add_argument("--train",type=str)
    parser.add_argument("--valid",type=str)
    parser.add_argument("--max_len",type=str,default='adam')
    parser.add_argument("--transformer_model",type=str,default='distilbert-base-uncased')
    

    return  parser.parse_known_args()


def _tokenize_data(x_train,x_valid,transformer_model):
    
    if transformer_model == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(transformer_model)


    x_train = X_train_processed.to_list()
    x_valid = X_valid_processed.to_list()


    train_encodings = tokenizer(x_train, max_length=MAX_LEN, truncation=True, padding='max_length',return_tensors='tf')
    valid_encodings = tokenizer(x_valid, max_length=MAX_LEN, truncation=True, padding='max_length',return_tensors='tf')
    
    return train_encodings,valid_encodings


def _encoding_labels(y_train,y_valid):
    
    print("Encoding Labels .....")
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

def _load_data(train_dir,valid_dir,MAX_LEN,epochs,batch_size,valid_batch_size,steps_per_epoch,validation_steps):
          
    print("train_dir : ",train_dir)    
    train_file = os.path.join(train_dir,"train.tfrecord") 
    print("train_file : ",train_file)
    
    print("valid_dir:",valid_dir)
    valid_file = os.path.join(valid_dir,"valid.tfrecord")
    print("valid_file:",valid_file)
    
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
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(_parse_function,num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    
    train_dataset.cache()
    
    valid_dataset = tf.data.TFRecordDataset(valid_file)
    valid_dataset = valid_dataset.repeat(epochs * validation_steps)
    valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.map(_parse_function,num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(valid_batch_size)
    
    valid_dataset.cache()
   
    return train_dataset, valid_dataset    
    
def main():
    args, unknown = _parse_args()
    print("input data: ",args.data_path)
    
    train, valid = _load_data()
    
    X_train, y_train  = None, None
    X_valid, y_valid  = None, None
    
    train_encodings, valid_encondings = _tokenize_data(X_train,X_valid,args.transformer_model)       
    y_train_encode, y_valid_encode = _encoding_labels(y_train,y_valid)
    
    train_tfdataset = construct_tfdataset(train_encodings, y_train_encode)
    valid_tfdataset = construct_tfdataset(valid_encodings, y_valid_encode)
    
    # training data
    _save_feature_as_tfrecord(train_tfdataset,'./data/train.tfrecord')

    #validation data
    _save_feature_as_tfrecord(valid_tfdataset,'./data/valid.tfrecord')


if __name__ == "__main__":
    main()