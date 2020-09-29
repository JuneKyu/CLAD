#!/usr/bin/env python
# -*- coding: utf-8 -*-

import config


def divide_data_label(dataset, train=False):
    in_data = []
    out_data = []
    in_labels = []
    out_labels = []
    for _d in dataset:
        data_x = _d[0].numpy()
        data_y = _d[1]

        #  if (data_y in config.normal_class_index_list):
        #      in_data.append(data_x)
        #      in_labels.append(0)
        #  else:
        #      in_data.append(data_x)
        #      in_labels.append(1)

        if (data_y in config.normal_class_index_list):
            in_data.append(data_x)
            in_labels.append(data_y)
        else:
            if (train):
                continue
            else:
                out_data.append(data_x)
                out_labels.append(data_y)

    return in_data, in_labels, out_data, out_labels
