#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import config

import pdb


def tpr95(name):
    # calculate the falsepositive error when tpr is 95%

    T = 1
    in_dist = np.loadtxt('./softmax_scores/')


def auroc(name):
    # calculate the AUROC

    T = 1
    in_dist = np.loadtxt(config.base_in_path, delimiter=',')
    out_dist = np.loadtxt(config.base_out_path, delimiter=',')

    # Base
    if name == 'mnist':
        start = 0.1
        end = 1
    else:
        start = 0.1
        end = 1

    gap = (end - start) / 100000
    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]
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

    if name == 'mnist':
        start = 0.1
        end = 0.12
        #  end = 0.2
    else:
        start = 0.1
        end = 1

    gap = (end - start) / 100000
    Y1 = out_dist[:, 2]
    X1 = in_dist[:, 2]
    aurocOdin = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        aurocOdin += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocOdin += fpr * tpr
    return aurocBase, aurocOdin


def calculate_metric(nn):

    log = config.logger
    aurocBase, aurocOdin = auroc(nn)
    print("auroc Base : {:.4f} , auroc Odin : {:.4f}".format(
        aurocBase, aurocOdin))
    log.info("auroc Base : {:.4f} , auroc Odin : {:.4f}".format(
        aurocBase, aurocOdin))
