import config
import numpy as np
import pandas as pd
import time
import datetime
import gc
import random
import nltk
from nltk.corpus import stopwords
import re
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
def preprocess_dataset(path):
    nltk.download('stopwords')
    sw = stopwords.words('english')
    df = pd.read_csv(path)

    df['Email Text'] = df['Email Text'].astype('str')
    df['Email Type'] = df['Email Type'].astype('str')
    mapping = {'Safe Email': 0, 'Phishing Email': 1}
    df['Email Type'] = df['Email Type'].map(mapping)
    df.rename(columns={'Email Type': 'label'}, inplace=True)
    df.rename(columns={'Email Text': 'text'}, inplace=True)

    #clean message bodies
    df['text'] = df['text'].apply(lambda x: clean_text(x, sw))
    return df


def clean_text(text,sw):

    # delete new lines and tabs
    text = text.replace("\n", " ").replace("\t", " ").strip()
    #lowercase
    text = text.lower()
    #change not used symbols to space
    text = re.sub(r"[^a-zA-Z?.!,$]+", " ", text)
    #remove links
    text = re.sub(r"http\S+", "",text)
    text = re.sub(r"http", "",text)
    text = re.sub(r"enron", "",text)
    #remove html tags
    html=re.compile(r'<.*?>')
    text = html.sub(r'',text)
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = " ".join(text)
    return text

#function to calculate the accuracy of predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def main():
    if config.intended_device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif config.intended_device == "cpu":
        device = torch.device("cpu")
    print("Using device: "+str(device))
    #preprocess
    df = preprocess_dataset(config.dataset_path)

    print("Loaded dataset, info:")
    print(df.info())

    emails = df.text.values
    labels = df.label.values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = []
    attention_masks = []

    # for each email tokenize, add [CLS] and [SEP] tokens, maps tokens to ids, pad or truncate and create attention masks for [PAD] tokens
    for email in emails:
        encoded_dict = tokenizer.encode_plus(
                            email,
                            add_special_tokens = True, #[CLS] and [SEP]
                            max_length = config.max_len, #pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   #attention masks.
                            return_tensors = 'pt',     # Return pytorch tensors
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    #convert lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    #convert to TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    #training/evaluating/testing sets size
    train_size = int(config.training_percent * len(dataset))
    val_size = int(config.testing_percent * len(dataset))

    #split randomly
    train_dataset, tmp_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.rand_seed))
    val_dataset, test_dataset_tmp = random_split(tmp_dataset, [int(config.training_percent * len(tmp_dataset)), int(config.testing_percent * len(tmp_dataset))], generator=torch.Generator().manual_seed(config.rand_seed))

    #DataLoaders for training and validation sets.
    #samples in random order.
    train_dataloader = DataLoader(
                train_dataset,
                sampler = RandomSampler(train_dataset),
                batch_size = config.batch_size
                )

    #for validation sequentially.
    validation_dataloader = DataLoader(
                val_dataset,
                sampler = SequentialSampler(val_dataset),
                batch_size = config.batch_size
                )

    #load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", #12-layer BERT model, with an uncased vocab
        num_labels = 2, #2 for binary classification.
        output_attentions = False, #whether the model returns attentions weights.
        output_hidden_states = False, #whether the model returns all hidden-states.
    )

    model = model.to(device)

    #optimizer
    optimizer = AdamW(model.parameters(),
                  lr = config.learning_rate,
                  eps = 1e-8 #default
                )

    #total number of training steps
    total_steps = len(train_dataloader) * config.epochs

    #learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    random.seed(config.rand_seed)
    np.random.seed(config.rand_seed)
    torch.manual_seed(config.rand_seed)
    torch.cuda.manual_seed_all(config.rand_seed)
    training_stats = []
    total_t0 = time.time()

    #for each epoch...
    for epoch_i in range(0, config.epochs):
        #training: perform one full pass over the training set.
        print("")
        print('Epoch {:} / {:} '.format(epoch_i + 1, config.epochs))
        print('Training...')
        #training time
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            #unpack training batch from our dataloader.
            #copy each tensor to the device using the `to` method.
            #batch contains three pytorch tensors: input_ids, attention_masks, labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            optimizer.zero_grad()
            output = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
            loss = output.loss
            total_train_loss += loss.item()
            #backward pass to calculate the gradients.
            loss.backward()
            #clip the norm of the gradients to 1.0. (to help prevent the "exploding gradients" problem)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            optimizer.step()
            #update the learning rate.
            scheduler.step()

        #calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        #epoch time
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        #Validation
        #measure our performance on validation set.
        print("")
        print("Validating...")
        t0 = time.time()
        #put the model in evaluation mode--the dropout layers behave differently during evaluation.
        model.eval()
        #tracking variables
        total_eval_accuracy = 0
        best_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        #evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            #tell pytorch not to bother with constructing the compute graph during the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                output= model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            loss = output.loss
            total_eval_loss += loss.item()
            #move logits and labels to CPU if we are using GPU
            logits = output.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            #calculate the accuracy for this batch of test sentences, and
            #accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        #report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        #calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        #measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        if avg_val_accuracy > best_eval_accuracy:
            torch.save(model, 'bert_model')
            best_eval_accuracy = avg_val_accuracy
        #record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    model = torch.load('bert_model')

    test_input_ids, test_attention_masks, test_labels = zip(*test_dataset_tmp)
    #create new TensorDataset without labels
    test_dataset = TensorDataset(torch.stack(test_input_ids), torch.stack(test_attention_masks))
    test_labels_list = [label.item() for label in test_labels]

    test_dataloader = DataLoader(
                test_dataset,
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = config.batch_size # Evaluate with this batch size.
            )

    predictions = []
    for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            with torch.no_grad():
                output= model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)
                logits = output.logits
                logits = logits.detach().cpu().numpy()
                pred_flat = np.argmax(logits, axis=1).flatten()

                predictions.extend(list(pred_flat))

    print("Test results: ")
    conf_matrix = confusion_matrix(test_labels_list, predictions)

    confusion_df = pd.DataFrame(
        conf_matrix,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )

    #display the labeled confusion matrix
    print("Confusion Matrix:")
    print(confusion_df)


    # generate report
    report = classification_report(test_labels_list, predictions, target_names=["Negative", "Positive"])
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main()
