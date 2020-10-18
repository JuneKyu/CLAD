#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import config

import pdb


def tpr95(name):
    # calculate the falsepositive error when tpr is 95%

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

    T = 1000
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


def auroc(name):
    # calculate the AUROC

    # TODO: adjust the length with test len

    T = 1
    in_dist = np.loadtxt(config.base_in_path, delimiter=',')
    out_dist = np.loadtxt(config.base_out_path, delimiter=',')

    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]

    # Base
    start = 0.1
    end = 1
    gap = (end - start) / 100000
    aurocBase = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        aurocBase += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocBase += fpr * tpr

    # Odin
    T = 1000
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
    gap = (end - start) / 100000
    #  if name == 'mnist':
    #      start = 0.1
    #      end = 1
    #      #  end = 0.2
    #  else:
    #      start = 0.1
    #      end = 1

    aurocOdin = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        aurocOdin += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocOdin += fpr * tpr
    return aurocBase, aurocOdin


def auprIn(name):
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

    # calculate odin algorithm
    T = 1000
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


def auprOut(name):
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

    # calculate odin algorithm
    T = 1000
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


def detection(name):
    # calculate the minimum detection error
    # calculate baseline
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

    # calculate odin algorithm
    T = 1000
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


def calculate_metric(nn):

    log = config.logger
    aurocBase, aurocOdin = auroc(nn)
    errorBase, errorOdin = detection(nn)
    #  tpr95Base, tpr95Odin = tpr95(nn)
    auprInBase, auprInOdin = auprIn(nn)
    auprOutBase, auprOutOdin = auprOut(nn)
    print("                   Base,     Odin")
    log.info("                   Base,     Odin")
    print("auroc : {:15.2f}   , {:3.2f}".format(aurocBase * 100,
                                                aurocOdin * 100))
    log.info("auroc : {:15.2f}   , {:3.2f}".format(aurocBase * 100,
                                                   aurocOdin * 100))
    print("detection error : {:3.2f} % , {:3.2f} %".format(
        errorBase * 100, errorOdin * 100))
    log.info("detection error : {:3.2f} % , {:3.2f} %".format(
        errorBase * 100, errorOdin * 100))
    #  print("fpr at tpr 95% : {:6.2f}   , {:3.2f}".format(
    #      tpr95Base * 100, tpr95Odin * 100))
    #  log.info("fpr at tpr 95% : {:6.2f}   , {:3.2f}".format(
    #      tpr95Base * 100, tpr95Odin * 100))
    print("aupr in : {:13.2f}   , {:3.2f}".format(auprInBase * 100,
                                                  auprInOdin * 100))
    log.info("aupr in : {:13.2f}   , {:3.2f}".format(auprInBase * 100,
                                                     auprInOdin * 100))
    print("aupr out : {:12.2f}   , {:3.2f}".format(auprOutBase * 100,
                                                   auprOutOdin * 100))
    log.info("aupr out : {:12.2f}   , {:3.2f}".format(auprOutBase * 100,
                                                      auprOutOdin * 100))
