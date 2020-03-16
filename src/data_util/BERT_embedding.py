#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import datetime
import random

import torch
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

from config import device
import config


def tokenize(tokenizer, sentences, MAX_LEN):

    input_ids = []

    for sent in sentences:
        encoded_sent = tokenizer.encode(
                sent,
                add_special_tokens=True
                )
        input_ids.append(encoded_sent)
    
    MAX_LEN = max([len(sen) for sen in input_ids])

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    return input_ids, MAX_LEN, attention_masks

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def fine_tune(train_data, val_data, test_data, num_labels, num_epochs, batch_size):

    log = config.logger

    log.info('================')
    log.info('BERT embedding model fine tuning : bert-base-uncase')
    log.info('================')

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    
    embedding_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = num_labels,
            output_attentions = False,
            output_hidden_states = True,
            )
    embedding_model.cuda()
    log.info('bert model num labels : ' + str(num_labels))
    
    optimizer = AdamW(embedding_model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )
    log.info('optimizer : AdamW with lr : 2e-5, eps : 1e-8')

    epochs = num_epochs

    total_steps = len(train_dataloader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    seed_val = 42
    log.info('seed value : ' + str(seed_val))
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch
    loss_values = []

    # ================
    # fine tuning
    # ================

    log.info("")
    log.info("================")
    log.info("fine tuning word dataset with BERT")
    log.info("================")

    print("")
    print("fine tuning word dataset with BERT")

    # For each epoch...
    for epoch_i in range(0, epochs):
        # ================
        # training
        # ================
        #  log.info("-"*99)

        print("")
        print("training bert embedding model...")

        log.info("training bert embedding model...")

        t0 = time.time()
        total_loss = 0 # reset total loss

        embedding_model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                log.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_attn_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            embedding_model.zero_grad()

            outputs = embedding_model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_attn_mask,
                                    labels=b_labels)

            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), 1.0) # prevent the "exploding gradient"

            optimizer.step()

            scheduler.step() # update the learning rate (in transformer architecture)

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        log.info("")
        log.info("  Average training loss: {0:.2f}".format(avg_train_loss))
        log.info("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
        # ================
        # validation
        # ================

        print("")
        print("running bert embedding validation...")
        log.info("")
        log.info("running bert embedding validation...")


        t0 = time.time()

        embedding_model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_step, nb_eval_examples = 0, 0

        for batch in val_dataloader:

            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_attn_mask, b_labels = batch

            with torch.no_grad():
                outputs = embedding_model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_attn_mask)
            
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy

            nb_eval_step += 1
        
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_step))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        log.info("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_step))
        log.info("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("fine tuning complete!")
    log.info("")
    log.info("fine tuning complete!")
        

    return 

#  def embedding_by_bert(train_data, val_data)

def preprocess_with_bert(dataset_name, train, val, test, num_labels = 2, MAX_LEN = 64, num_epochs = 5, batch_size = 32):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

    train_inputs, train_MAX_LEN, train_attn_masks = tokenize(tokenizer, train.sentence.values, MAX_LEN)
    train_labels = train.label.values

    val_inputs, val_MAX_LEN, val_attn_masks = tokenize(tokenizer, val.sentence.values, MAX_LEN)
    val_labels = val.label.values

    test_inputs, test_MAX_LEN, test_attn_masks = tokenize(tokenizer, test.sentence.values, MAX_LEN)
    test_labels = test.label.values
    
    # convert to Pytorch data types
    train_inputs = torch.tensor(train_inputs)
    train_attn_masks = torch.tensor(train_attn_masks)
    train_labels = torch.tensor(train_labels)
    val_inputs = torch.tensor(val_inputs)
    val_attn_masks = torch.tensor(val_attn_masks)
    val_labels = torch.tensor(val_labels)
    test_inputs = torch.tensor(test_inputs)
    test_attn_masks = torch.tensor(test_attn_masks)
    test_labels = torch.tensor(test_labels)

    # Create the DataLoader
    train_data = TensorDataset(train_inputs, train_attn_masks, train_labels)
    val_data = TensorDataset(val_inputs, val_attn_masks, val_labels)
    test_data = TensorDataset(test_inputs, test_attn_masks, test_labels)
    
    # fine tune
    fine_tune(train_data = train_data, val_data = val_data, test_data = test_data, num_labels = num_labels, num_epochs = num_epochs, batch_size = batch_size)

    train_x = train_inputs
    train_y = train_labels
    val_x = val_inputs
    val_y = val_labels
    test_x = test_inputs
    test_y = test_labels

    return train_x, train_y, val_x, val_y, test_x, test_y


