#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random

import torch
from transformers import BertTokenizer
from transformers import BertModel
#  from keras_preprocessing.sequence import pad_sequences

from sentence_transformers import SentenceTransformer

from config import device
import config

import pdb

#  def bert_tokenize(sentences):
#      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
#      tokenized_sentences = []
#      for sent in sentences:
#          tokens = torch.tensor(tokenizer.encode(sent)).unsqueeze(0)
#          tokenized_sentences.append(tokens)
#
#      pdb.set_trace()
#      tokenized_sentences = torch.stack(tokenized_sentences,1)
#      return tokenized_sentences


def avg_bert_embed(sentences):
    embedding_model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)

    for i, sent in enumerate(sentences):

        if (i + 1) % 100 == 0:
            print("embedding " + str(i + 1) + " out of " + str(len(sentences)))

        tokens = tokenizer.encode(sent)
        tokens = torch.tensor(tokens).unsqueeze(0).to(device)
        embedding_model.cuda()
        bert_embeddings = embedding_model(tokens)[0][
            0]  # get the embedding from the pretrained bert model

        num_tokens = len(bert_embeddings)
        embedding_size = len(bert_embeddings[0])
        avg_bert = torch.zeros([embedding_size],
                               dtype=torch.float64).to(device)

        for index in range(embedding_size):
            for token in bert_embeddings:
                avg_bert[index] += token[index].item()
            avg_bert[index] = avg_bert[index] / num_tokens

        # add as a numpy array
        embeddings.append(avg_bert.cpu().numpy())

    return embeddings


def sentence_bert_embed(sentences):

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(sentences)

    return sentence_embeddings


#  def preprocess_with_avg_glove()


def preprocess_with_avg_bert(train, test):

    train_x = avg_bert_embed(train.sentence.values)
    train_y = train.label.values
    #  val_x = torch.tensor(avg_bert_embed(val.sentence.values))
    #  val_y = torch.tensor(val.label.values)
    test_x = avg_bert_embed(test.sentence.values)
    test_y = test.label.values

    return train_x, train_y, test_x, test_y


def preprocess_with_s_bert(train, test):

    #  pdb.set_trace()
    train_x = sentence_bert_embed(train.sentence.values)
    train_y = train.label.values
    #  val_x = torch.tensor(sentence_bert_embed(val.sentence.values))
    #  val_y = torch.tensor(val.label.values)
    test_x = sentence_bert_embed(test.sentence.values)
    test_y = test.label.values

    return train_x, train_y, test_x, test_y
