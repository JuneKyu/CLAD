import numpy as np


# fuction that make start & end time list

def make_start_end_list(test_y):
    attack_time_list = []
    check = 0

    for i in range(test_y.shape[0]):
        if check == 0 and test_y[i] == 1:
            attack_time_list.append(i)
            check = 1
        elif check == 1 and test_y[i] == 0:
            attack_time_list.append(i)
            check = 0

    return attack_time_list


# find attack num

def find_attack_num(prediction, test_y):
    attack_list = []

    start_end_list = make_start_end_list(test_y)
    
    for attack_idx in range(0, len(start_end_list), 2):
        shapes, counts = np.unique(prediction[start_end_list[attack_idx] : \
                                    start_end_list[attack_idx+1]],
                                    return_counts=True)
        if counts.shape[0] > 1:
            attack_list.append(counts[1]/(counts[0] + counts[1]))
        elif shapes == 1:
            attack_list.append(1.0)
        elif shapes == 0:
            attack_list.append(0.0)

    return 35 - np.sum((np.array(attack_list) != 0.0).astype(int))
            

















