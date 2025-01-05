import config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
import time
import datetime
import gc
import random
import nltk
import re
import ssl
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup


if (config.PARAM1):
    device = torch.device("cpu")
else:
    "cuda:0" if torch.cuda.is_available() else torch.device("cpu")

print(device)


# This part is responsible for importing and preparing data
df_training = pd.read_csv(config.PARAM2)
df_test = pd.read_csv(config.PARAM3)
# Display the dataframe
#print(df_training.info())
#print(df_test.info())


# this part is downloading stopwoards to clean dataset before learning
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')


from nltk.corpus import stopwords
sw = stopwords.words('english')

def clean_text(text):
    
    text = text.lower()
    
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

    text = re.sub(r"http\S+", "",text) #Removing URLs 
    #text = re.sub(r"http", "",text)
    
    html=re.compile(r'<.*?>') 
    
    text = html.sub(r'',text) #Removing html tags
    
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,'') #Removing punctuations
        
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    
    text = " ".join(text) #removing stopwords
    return text

df_training['text'] = df_training['text'].astype('str')
df_training['text'] = df_training['text'].apply(lambda x: clean_text(x))

# check class distribution
print(df_training['label'].value_counts(normalize = True))


train_text, val_text, train_labels, val_labels = train_test_split(df_training['text'], df_training['label'],random_state=2018,test_size=config.PARAM4,stratify=df_training['label'])


# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# sample data
text = ["this is a bert model tutorial", "we will fine-tune a bert model"]

# encode text
sent_id = tokenizer.batch_encode_plus(text, padding=True, return_token_type_ids=False)

# output
print(sent_id)

# get length of all the messages in the train set
seq_len = [len(i.split()) for i in train_text]
hist, bin_edges = np.histogram(seq_len, bins=30)
hist_table = pd.DataFrame({'Message lenght': [f"{int(bin_edges[i])} - {int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],'Frequency': hist})
print(hist_table)
