# import config and class and function
import config
from BERT_Arch_Class import BERT_Arch

# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
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

# this part is downloading stopwoards to clean dataset before learning
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
from nltk.corpus import stopwords

if (config.PARAM1 == "true"):
    device = torch.device("cpu")
else:
    device = "cuda:0" 
print(device)

# This part is responsible for importing and preparing data
df_training = pd.read_csv(config.PARAM2)
df_test = pd.read_csv(config.PARAM3)

sw = stopwords.words('english')

def clean_text(text, clean_stopwords):
    text = text.lower() # because using bert-base-uncased
    text = re.sub(r"http\S+", "",text) #Removing URLs 
    text = re.sub(r"[^a-zA-Z?.!,¿$]+", " ", text) #Removing special characters
    html=re.compile(r'<.*?>') 
    text = html.sub(r'',text) #Removing html tags
    if clean_stopwords:
        text = [word for word in text.split() if word not in sw]
    text = " ".join(text) #removing stopwords
    return text

df_training['text'] = df_training['text'].astype('str')
df_training['text'] = df_training['text'].apply(lambda x: clean_text(x, config.PARAM6))

# check class distribution
print(df_training['label'].value_counts(normalize = True))

# prepare training set and validation set
train_text, val_text, train_labels, val_labels = train_test_split(
    df_training['text'], 
    df_training['label'],
    random_state=2018,
    test_size=config.PARAM4,
    stratify=df_training['label'])

# prepare test set
test_text = df_test['text']  # Features (email text)
test_text = test_text.astype('str')
test_labels = df_test['label']  # Labels (e.g., ham/spam or 0/1)

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# get length of all the messages in the train set
seq_len = [len(i.split()) for i in train_text]
hist, bin_edges = np.histogram(seq_len, bins=len(seq_len))
hist_table = pd.DataFrame({'Message lenght': [f"{int(bin_edges[i])} - {int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],'Frequency': hist})
cumulative_frequency = np.cumsum(hist)
# Total frequency
total_frequency = cumulative_frequency[-1]
# Find the bin edge that covers the desired percentage
cutoff_frequency = total_frequency * 0.9
bin_index = np.searchsorted(cumulative_frequency, cutoff_frequency)

# Return the upper edge of the bin covering the specified percentage
if int(bin_index)+1 > 512:
    max_seq_len = 512
else:
   max_seq_len = int(bin_edges[bin_index + 1])
print(max_seq_len)

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = max_seq_len,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = max_seq_len,
    padding="max_length",
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
   test_text.tolist(),
   max_length = max_seq_len,
   padding="max_length",
   truncation=True,
   return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

#define a batch size
batch_size = config.PARAM7
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)
# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)
# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False


# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)
# push the model to GPU or CPU as selected in the begining
model = model.to(device)
# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)

class_wts = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
print(class_wts)

# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)
# loss function
cross_entropy  = nn.NLLLoss(weight=weights)
# number of training epochs
epochs = config.PARAM5

def format_time(seconds):
    """Convert seconds into a formatted string (e.g., HH:MM:SS)."""
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{int(hours):02}:{int(mins):02}:{int(secs):02}"

# function for evaluating the model
def evaluate():

  print("\nEvaluating...")

  # deactivate dropout layers
  model.eval()
  total_loss, total_accuracy = 0, 0
  # empty list to save the model predictions
  total_preds = []
  t0 = time.time()
  # iterate over batches
  for step,batch in enumerate(val_dataloader):
    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
      # Calculate elapsed time in minutes.
      elapsed = format_time(time.time() - t0)
      t0 = time.time()
      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
      print(elapsed)
    # push the batch to device
    batch = [t.to(device) for t in batch]
    sent_id, mask, labels = batch
    # deactivate autograd
    with torch.no_grad():
      # model predictions
      preds = model(sent_id, mask)
      # compute the validation loss between actual and predicted values
      loss = cross_entropy(preds,labels)
      total_loss = total_loss + loss.item()
      preds = preds.detach().cpu().numpy()
      total_preds.append(preds)
  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader)
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)
  return avg_loss, total_preds

# function to train the model
def train():
  model.train()
  total_loss, total_accuracy = 0, 0
  # empty list to save model predictions
  total_preds=[]
  # iterate over batches
  for step,batch in enumerate(train_dataloader):
    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
    # push the batch to device
    batch = [r.to(device) for r in batch]
    sent_id, mask, labels = batch
    # clear previously calculated gradients
    model.zero_grad()
    # get model predictions for the current batch
    preds = model(sent_id, mask)
    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)
    # add on to the total loss
    total_loss = total_loss + loss.item()
    # backward pass to calculate the gradients
    loss.backward()
    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # update parameters
    optimizer.step()
    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()
    # append the model predictions
    total_preds.append(preds)
  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)
  #returns the loss and predictions
  return avg_loss, total_preds

# set initial loss to infinite
best_valid_loss = float('inf')
# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

if config.PARAM10:
    #for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        #train model
        train_loss, _ = train()
        #evaluate model
        valid_loss, _ = evaluate()
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')


if config.PARAM11:
    #load weights of best model
    path = 'saved_weights.pt'
    model.load_state_dict(torch.load(path))
    # get predictions for test data
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()

    # model's performance
    preds = np.argmax(preds, axis = 1)
    print(classification_report(test_y, preds))

    # confusion matrix
    pd.crosstab(test_y, preds)