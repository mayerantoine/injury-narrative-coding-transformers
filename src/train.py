import os
import tensorflow as tf
import numpy as np
import argparse
from transformers import  TFDistilBertForSequenceClassification,DistilBertConfig
from transformers import TFBertForSequenceClassification, BertConfig
from transformers import TFRobertaForSequenceClassification, RobertaConfig
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from transformers.optimization_tf import WarmUp, AdamWeightDecay
from pickle import load

def _parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model-dir",type=str)
    #parser.add_argument('--output-data-dir', type=str,default=os.environ['SM_OUTPUT_DATA_DIR'])
    #parser.add_argument("--sm-model-dir",type=str,default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train",type=str,default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation",type=str,default=os.environ.get("SM_CHANNEL_VALIDATION"))
    
    ## Hyperparams
    parser.add_argument("--model_name",type=str,default='BaseBERT')
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

class DistilBERT:
    def __init__(self, params):
        self.num_labels = params['num_labels']
        self.max_len = params['max_len']
        self.learning_rate = params['learning_rate']
        self.id2label = params['id2label']
        self.label2id = params['label2id']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.num_records = params['num_records']
        
    def build(self):
        
        freeze_bert_layer = False
        
        
        transformer_config = DistilBertConfig(num_labels=self.num_labels,
                                   id2label = self.id2label,
                                   label2id  = self.label2id,
                                   output_attentions = False,
                                   output_hidden_states= False)
                                   
        transformer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",config=transformer_config)
        input_ids = tf.keras.Input(shape = (self.max_len,),name='input_ids',dtype='int32')
        attention_mask = tf.keras.Input(shape = (self.max_len,),name='attention_mask',dtype='int32')
        
        embbeding_layer = transformer_model.distilbert(input_ids,attention_mask=attention_mask)[0]
        X = tf.keras.layers.Conv1D(512, 3, activation='relu')(embbeding_layer)
        X = tf.keras.layers.GlobalMaxPool1D()(X)
        X = tf.keras.layers.Dense(256, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.2)(X)
        X = tf.keras.layers.Dense(self.num_labels, activation='softmax')(X)
        
        model = tf.keras.Model(inputs=[input_ids,attention_mask], outputs = X)
        
        for layer in model.layers[:3]:
            layer.trainable = not freeze_bert_layer
            
        num_train_steps = (self.num_records // self.batch_size) * self.epochs
        decay_schedule_fn = PolynomialDecay(
            initial_learning_rate=self.learning_rate,
            end_learning_rate=0.,
            decay_steps=num_train_steps
            )
        
        warmup_steps = num_train_steps * 0.1
        warmup_schedule  = WarmUp(self.learning_rate,decay_schedule_fn,warmup_steps)
        
        # fine optimizer and loss
        #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        optimizer = AdamWeightDecay (learning_rate=warmup_schedule, weight_decay_rate=0.01, epsilon=1e-6)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= False)
        metrics = ['acc']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

class BaseBERT:
    def __init__(self, params):
        self.num_labels = params['num_labels']
        self.max_len = params['max_len']
        self.learning_rate = params['learning_rate']
        self.id2label = params['id2label']
        self.label2id = params['label2id']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.num_records = params['num_records']
        
    def build(self):      
        
        freeze_bert_layer = False
        
        transformer_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=self.num_labels, 
                 output_attentions = False,
                output_hidden_states= False )
        input_ids = tf.keras.Input(shape = (self.max_len,),name='input_ids',dtype='int32')
        attention_mask = tf.keras.Input(shape = (self.max_len,),name='attention_mask',dtype='int32')
        
        embbeding_layer = transformer_model.bert(input_ids,attention_mask=attention_mask)[0]
        X = tf.keras.layers.Conv1D(512, 3, activation='relu')(embbeding_layer)
        X = tf.keras.layers.GlobalMaxPool1D()(X)
        X = tf.keras.layers.Dense(256, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.2)(X)
        X = tf.keras.layers.Dense(self.num_labels, activation='softmax')(X)

        model = tf.keras.Model(inputs=[input_ids,attention_mask], outputs = X)
        
        for layer in model.layers[:3]:
            layer.trainable = not freeze_bert_layer
        
        num_train_steps = (self.num_records // self.batch_size) * self.epochs
        decay_schedule_fn = PolynomialDecay(
            initial_learning_rate=self.learning_rate,
            end_learning_rate=0.,
            decay_steps=num_train_steps
            )
        
        warmup_steps = num_train_steps * 0.1
        warmup_schedule  = WarmUp(self.learning_rate,decay_schedule_fn,warmup_steps)
        
        #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        optimizer = AdamWeightDecay (learning_rate=warmup_schedule, weight_decay_rate=0.01, epsilon=1e-6)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        
        return model
    

class BaseRoberta:
    
    def __init__(self, params):
        self.num_labels = params['num_labels']
        self.max_len = params['max_len']
        self.learning_rate = params['learning_rate']
        self.id2label = params['id2label']
        self.label2id = params['label2id']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.num_records = params['num_records']
        
    def build(self):      
        
        freeze_bert_layer = False
        
        transformer_model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=self.num_labels, 
                 output_attentions = False,
                output_hidden_states= False )
        input_ids = tf.keras.Input(shape = (self.max_len,),name='input_ids',dtype='int32')
        attention_mask = tf.keras.Input(shape = (self.max_len,),name='attention_mask',dtype='int32')
        
        embbeding_layer = transformer_model.roberta(input_ids,attention_mask=attention_mask)[0]
        X = tf.keras.layers.GlobalMaxPool1D()(embbeding_layer)
        X = tf.keras.layers.Dense(self.num_labels, activation='softmax')(X)
        
        model = tf.keras.Model(inputs=[input_ids,attention_mask], outputs = X)
        
        for layer in model.layers[:3]:
            layer.trainable = not freeze_bert_layer
        
        num_train_steps = (self.num_records // self.batch_size) * self.epochs
        decay_schedule_fn = PolynomialDecay(
            initial_learning_rate=self.learning_rate,
            end_learning_rate=0.,
            decay_steps=num_train_steps
            )
        
        warmup_steps = num_train_steps * 0.1
        warmup_schedule  = WarmUp(self.learning_rate,decay_schedule_fn,warmup_steps)
        
        #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        optimizer = AdamWeightDecay (learning_rate=warmup_schedule, weight_decay_rate=0.01, epsilon=1e-6)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        
        return model
        
        
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

def _load_encoder(train_dir):
    # load the encoder
    encoder_file = os.path.join(train_dir,'encode.pkl') 
    encoder = load(open(encoder_file, 'rb'))
    
    return encoder

def main():
    args, unknown = _parse_args()
    print("input train: ",args.train)
    print("input valid: ",args.validation)

    train_dataset, valid_dataset = _load_data(args.train,
                                              args.validation,
                                              args.max_len,
                                              args.epochs,
                                              args.batch_size,
                                              args.valid_batch_size,
                                              args.steps_per_epoch,
                                              args.validation_steps)    
    encoder = _load_encoder(args.train)
    
    CLASSES = encoder.classes_
    id2label = { id:str(label) for id, label in enumerate(encoder.classes_)}
    label2id = { str(label):id for id, label in enumerate(encoder.classes_)}
    
    params = {
            'num_labels': len(CLASSES),
            'max_len': args.max_len,
            'learning_rate': args.learning_rate,
            'id2label': id2label,
            'label2id': label2id,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'num_records':args.num_records
        }
    
    if args.model_name == 'DistilBERT' :
        model = DistilBERT(params).build()   
    elif args.model_name == 'BaseBERT':
        model = BaseBERT(params).build()
    elif args.model_name == 'BaseRoberta':
        model = BaseRoberta(params).build()
        
    print(model.summary())
    
    print("validation steps:",args.validation_steps)
        
    print("Training....")
    history = model.fit(x=train_dataset.shuffle(args.num_records),
                        steps_per_epoch = args.steps_per_epoch,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        validation_data=valid_dataset,
                        validation_steps=args.validation_steps)
        
    
    print("saving model...")
    #model.save(os.path.join(args.sm_model_dir,f"{args.model_name}.h5"))
    model.save(os.path.join(args.sm_model_dir,f"{args.model_name}"))
    #odel.save(f"./output/model/{args.model_name}/{args.model_name}_test.h5")

    
if __name__ == "__main__":
    main()




