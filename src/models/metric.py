#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import config

import pdb


def tpr95():
    # calculate the falsepositive error when tpr is 95%

    # Base
    T = 1
    in_dist = np.loadtxt(config.base_in_path, delimiter=',')
    out_dist = np.loadtxt(config.base_out_path, delimiter=',')

    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]

    start = 0.1
    end = 1
    gap = (end - start) / 100000
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            #  if tpr <= 0.955 and tpr >= 0.945:
            fpr += error2
            total += 1
    fprBase = fpr / total

    # Odin
    T = config.temperature
    in_dist = np.loadtxt(config.odin_in_path, delimiter=',')
    out_dist = np.loadtxt(config.odin_out_path, delimiter=',')

    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]

    start_Y = np.min(Y1)
    start_X = np.min(X1)
    start = min(start_Y, start_X)
    end_Y = np.max(Y1)
    end_X = np.max(X1)
    end = max(end_Y, end_X)

    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 > delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            #  if tpr <= 0.955 and tpr >= 0.945:
            fpr += error2
            total += 1
    fprOdin = fpr / total

    return fprBase, fprOdin


def f1():

    # Base
    T = 1
    in_dist = np.loadtxt(config.base_in_path, delimiter=',')
    out_dist = np.loadtxt(config.base_out_path, delimiter=',')

    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]

    scores = np.append(X1, Y1)
    labels = []
    for i in range(len(scores)):
        if (i < len(X1)):
            labels.append(0)
        else:
            labels.append(1)
    precision, recall, thresholds = precision_recall_curve(labels,
                                                           scores,
                                                           pos_label=0)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1Base = max(f1)

    # Odin
    T = config.temperature
    in_dist = np.loadtxt(config.odin_in_path, delimiter=',')
    out_dist = np.loadtxt(config.odin_out_path, delimiter=',')

    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]

    scores = np.append(X1, Y1)
    labels = []
    for i in range(len(scores)):
        if (i < len(X1)):
            labels.append(0)
        else:
            labels.append(1)
    precision, recall, thresholds = precision_recall_curve(labels,
                                                           scores,
                                                           pos_label=0)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1Odin = max(f1)

    return f1Base, f1Odin


def auroc():
    # calculate the AUROC
    # TODO: adjust the length with test len

    # Base
    T = 1
    in_dist = np.loadtxt(config.base_in_path, delimiter=',')
    out_dist = np.loadtxt(config.base_out_path, delimiter=',')

    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]

    scores = np.append(X1, Y1)
    labels = []
    for i in range(len(scores)):
        if (i < len(X1)):
            labels.append(0)
        else:
            labels.append(1)
    np.save(os.path.join(config.sub_log_path, "labels_base.txt"), labels)
    np.save(os.path.join(config.sub_log_path, "scores_base.txt"), scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
    aurocBase = auc(fpr, tpr)
    if aurocBase < 0.5: aurocBase = 1-aurocBase

    # Odin
    T = config.temperature
    in_dist = np.loadtxt(config.odin_in_path, delimiter=',')
    out_dist = np.loadtxt(config.odin_out_path, delimiter=',')

    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]

    scores = np.append(X1, Y1)
    labels = []
    for i in range(len(scores)):
        if (i < len(X1)):
            labels.append(0)
        else:
            labels.append(1)
    np.save(os.path.join(config.sub_log_path, "labels_odin.txt"), labels)
    np.save(os.path.join(config.sub_log_path, "scores_odin.txt"), scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
    aurocOdin = auc(fpr, tpr)
    if aurocOdin < 0.5: aurocOdin = 1-aurocOdin

    return aurocBase, aurocOdin


def auprIn():

    # Base
    T = 1
    in_dist = np.loadtxt(config.base_in_path, delimiter=',')
    out_dist = np.loadtxt(config.base_out_path, delimiter=',')
    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]
    start = 0.1
    end = 1
    gap = (end - start) / 100000
    precisionVec = []
    recallVec = []
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    # Odin
    T = config.temperature
    in_dist = np.loadtxt(config.odin_in_path, delimiter=',')
    out_dist = np.loadtxt(config.odin_out_path, delimiter=',')
    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]
    start_Y = np.min(Y1)
    start_X = np.min(X1)
    start = min(start_Y, start_X)
    end_Y = np.max(Y1)
    end_X = np.max(X1)
    end = max(end_Y, end_X)
    auprOdin = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        auprOdin += (recallTemp - recall) * precision
        recallTemp = recall
    auprOdin += recall * precision
    return auprBase, auprOdin


def auprOut():

    # Base
    T = 1
    in_dist = np.loadtxt(config.base_in_path, delimiter=',')
    out_dist = np.loadtxt(config.base_out_path, delimiter=',')
    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]
    start = 0.1
    end = 1
    gap = (end - start) / 100000
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    # Odin
    T = config.temperature
    in_dist = np.loadtxt(config.odin_in_path, delimiter=',')
    out_dist = np.loadtxt(config.odin_out_path, delimiter=',')
    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]
    start_Y = np.min(Y1)
    start_X = np.min(X1)
    start = min(start_Y, start_X)
    end_Y = np.max(Y1)
    end_X = np.max(X1)
    end = max(end_Y, end_X)
    auprOdin = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprOdin += (recallTemp - recall) * precision
        recallTemp = recall
    auprOdin += recall * precision
    return auprBase, auprOdin


def detection():
    # calculate the minimum detection error

    # Base
    T = 1
    in_dist = np.loadtxt(config.base_in_path, delimiter=',')
    out_dist = np.loadtxt(config.base_out_path, delimiter=',')
    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]
    start = 0.1
    end = 1
    gap = (end - start) / 100000
    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    # Odin
    T = config.temperature
    in_dist = np.loadtxt(config.odin_in_path, delimiter=',')
    out_dist = np.loadtxt(config.odin_out_path, delimiter=',')
    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]
    start_Y = np.min(Y1)
    start_X = np.min(X1)
    start = min(start_Y, start_X)
    end_Y = np.max(Y1)
    end_X = np.max(X1)
    end = max(end_Y, end_X)
    errorOdin = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorOdin = np.minimum(errorOdin, (tpr + error2) / 2.0)

    return errorBase, errorOdin


def calculate_metric():

    log = config.logger
    f1Base, f1Odin = f1()
    aurocBase, aurocOdin = auroc()
    #  errorBase, errorOdin = detection()
    #  tpr95Base, tpr95Odin = tpr95()
    #  auprInBase, auprInOdin = auprIn()
    #  auprOutBase, auprOutOdin = auprOut()
    print("{:>21}{:>13}".format("Base", "Odin"))
    log.info("{:>21}{:>13}".format("Base", "Odin"))
    print("")
    log.info("")
    print("{:14}{:7.2f}%{:>12.2f}%".format("F1:", f1Base * 100, f1Odin * 100))
    log.info("{:14}{:7.2f}%{:>12.2f}%".format("F1:", f1Base * 100,
                                              f1Odin * 100))
    print("{:14}{:7.2f}%{:>12.2f}%".format("AUROC:", aurocBase * 100,
                                           aurocOdin * 100))
    log.info("{:14}{:7.2f}%{:>12.2f}%".format("AUROC:", aurocBase * 100,
                                              aurocOdin * 100))
    #  print("{:14}{:7.2f}%{:>12.2f}%".format("DETECTION:", errorBase * 100,
    #                                         errorOdin * 100))
    #  log.info("{:14}{:7.2f}%{:>12.2f}%".format("DETECTION:", errorBase * 100,
    #                                            errorOdin * 100))
    #  print("{:14}{:7.2f}%{:>12.2f}%".format("AUPR IN:", auprInBase * 100,
    #                                         auprInOdin * 100))
    #  log.info("{:14}{:7.2f}%{:>12.2f}%".format("AUPR IN:", auprInBase * 100,
    #                                            auprInOdin * 100))
    #  print("{:14}{:7.2f}%{:>12.2f}%".format("AUPR OUT:", auprOutBase * 100,
    #                                         auprOutOdin * 100))
    #  log.info("{:14}{:7.2f}%{:>12.2f}%".format("AUPR OUT:", auprOutBase * 100,
    #                                            auprOutOdin * 100))
    #  print("")
