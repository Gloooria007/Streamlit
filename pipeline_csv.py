#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import re
# model 
import random
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader

# user defined parameter
batch_size = 4
max_length = 512
seed = 42
stop_words = ['a','an','the']
ORGMODEL = "D:/Python/jupyter/0.capstone/streamlit/Model/finetuned_bert_latest_epoch.model"
bert_path = 'D:/Python/jupyter/0.capstone/streamlit/Model'


# In[28]:


#####define data preprocessing function##############
def preprocess_text(text):

    #remove url
    text = re.sub(r"http\S+", "", text)

    #remove special characters
    text = re.sub('\W', ' ', text)

    #remove digits
    text = re.sub(r'\d+', '', text)

    #remove _
    text = text.replace('_',' ')

    #remove white spaces
    text = re.sub(r"\s+"," ", text)

    #to lower
    text = text.lower()
    
    # clean filler
    text =' '.join([word for word in text.split() if 'filler' not in word])

    #remove stopwords
    text =' '.join([word for word in text.split() if word not in (stop_words)])
    
    return text


def clean_text(text):
    
    #remove repeated punctuation 
    text = re.sub(r'(\W)(?=\1)', '',text)
    
    # clean !? into single one
    text = re.sub(r'[!?]{2,}','?!',text)

    #remove url
    text = re.sub(r"http\S+", "", text)

    #remove special characters except punctuations
    text = re.sub('[^A-Za-z0-9.,!?:;"]+', ' ', text)

    #remove _
    text = text.replace('_',' ')

    #remove white spaces
    text = re.sub(r"\s+"," ", text)

    #to lower
    text = text.lower()
    
    # clean filler
    text =' '.join([word for word in text.split() if 'filler' not in word])

    #remove stopwords
    text =' '.join([word for word in text.split() if word not in (stop_words)])
    
    processed_text = preprocess_text(text)
    
    # remove longwords
    pure_word_count = len(processed_text.split(' '))
    pure_char_count = sum(len(word) for word in str(processed_text).split(" "))
    avg_word_length = pure_char_count / pure_word_count
    
    if avg_word_length>=15:
        text =' '.join([word for word in text.split() if len(word)<15])
    
    return text


# In[29]:


# Define model
# Define evaluate & predict function 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.empty_cache()
 
# Set device
# torch.cuda.get_device_name(0)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# model predict function
def predict(df):
    # tokenizing text
    tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)
    
    encoded_data_test = tokenizer.batch_encode_plus(
    df['cleaned_text'].values.tolist(), 
    add_special_tokens=True, 
    # add_special_tokens=False, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=max_length,
    truncation=True, 
    return_tensors='pt'
    )
    
    # bert input
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    
    # tensor dataset
    dataset_test = TensorDataset(input_ids_test, attention_masks_test)
    
    # dataloader
    dataloader_test = DataLoader(dataset_test, 
                batch_size=batch_size)
    
    
    # load model
    model = BertForSequenceClassification.from_pretrained(bert_path,
                                                          num_labels=2,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.to(device)
    model.load_state_dict(torch.load(ORGMODEL, map_location=torch.device(device))['state_dict'])
    
    model.eval()

    predict_probas= []
    
    for batch in dataloader_test:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1] 
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            predict_proba = F.softmax(outputs.logits, dim=1)
            
        predict_proba = predict_proba.detach().cpu().numpy()
        predict_probas.append(predict_proba[:,1])    

    predict_probas = np.concatenate(predict_probas, axis=0)
            
    return predict_probas


# In[30]:


####### define pipeline ########
def run_pipeline(df):
   
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    test_predict_probas = predict(df)
    
    return test_predict_probas




