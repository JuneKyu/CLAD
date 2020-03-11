import sys, getopt
import time, datetime
from typing import Callable
import math


# To store a single anomaly
class Term:
    def __init__(self, first, last, name):
        self._first_timestamp = first
        self._last_timestamp = last
        self._name = name

    def set_time(self, first, last):
        self._first_timestamp = first
        self._last_timestamp = last

    def get_time(self):
        return self._first_timestamp, self._last_timestamp

    def set_name(self, str):
        self._name = str

    def get_name(self):
        return self._name

    def __eq__(self, other):
        return self._first_timestamp == other.get_time()[0] and self._last_timestamp == other.get_time()[1]
    
    
class TaPR:
    def __init__(self, label, theta, delta):
        self._predictions = []  # list of Terms
        self._anomalies = []    # list of Terms
        self._ambiguous_inst = [] # list of Terms

        self._set_predictions = False
        self._set_anomalies = False

        assert(len(label) == 2)
        self._normal_lbl = label[0]
        self._anomal_lbl = label[1]

        self._theta = theta
        self._delta = delta
        pass

    def load_predictions(self, filename):
        ntoken = self._check_file_format(filename)

        if ntoken == 1:
            self._predictions = self._load_timeseries_file(filename)
        else:
            self._predictions = self._load_range_file(filename)
        self._set_prediction = True


    def load_anomalies(self, filename):
        ntoken = self._check_file_format(filename)

        if ntoken == 1:
            self._anomalies = self._load_timeseries_file(filename)
        else:
            self._anomalies = self._load_range_file(filename)
        self._set_anomalies = True

        self._gen_ambiguous()


    def _gen_ambiguous(self):
        for i in range(len(self._anomalies)):
            start_id = self._anomalies[i].get_time()[1] + 1
            end_id = start_id + self._delta -1

            #if the next anomaly occurs during the theta, update the end_id
            if i+1 < len(self._anomalies) and end_id > self._anomalies[i+1].get_time()[0]:
                end_id = self._anomalies[i+1].get_time()[0]

            self._ambiguous_inst.append(Term(start_id, end_id, str(i)))


    def _check_file_format(self, filename):
        # check the file's format
        f = open(filename, 'r', encoding='utf-8', newline='')
        line = f.readline()
        token = line.strip().split(',')
        f.close()
        return len(token)

    def _load_range_file(self, filename):
        temp_list = []
        f = open(filename, 'r', encoding='utf-8', newline='')
        for line in f.readlines():
            items = line.strip().split(',')
            if len(items) > 2:
                temp_list.append(Term(int(items[0]), int(items[1]), str(items[2])))
            else:
                temp_list.append(Term(int(items[0]), int(items[1]), 'undefined'))
        f.close()
        return temp_list

    def _load_timeseries_file(self, filename):
        return_list = []
        start_id = -1
        id = 0
        range_id = 1
        #set prev_val as a value different to normal and anomalous labels
        prev_val = self._anomal_lbl-1
        if prev_val == self._normal_lbl:
            prev_val -= 1

        f = open(filename, 'r', encoding='utf-8', newline='')
        for line in f.readlines():
            val = int(line.strip().split()[0])

            if val == self._anomal_lbl and prev_val == self._normal_lbl:
                start_id = id
            elif val == self._normal_lbl and prev_val == self._anomal_lbl:
                return_list.append(Term(start_id, id - 1, str(range_id)))
                range_id += 1
                start_id = 0
            elif start_id == -1 and val == self._anomal_lbl:
                start_id = 0

            id += 1
            prev_val = val
        f.close()
        if start_id != 0:
            return_list.append(Term(start_id, id-1, str(range_id)))

        return return_list


    def get_n_predictions(self):
        return len(self._predictions)

    def get_n_anomalies(self):
        return len(self._anomalies)

    # return a value with the detected anomaly list
    def TaR_d(self) -> {float, list}:
        total_score = 0.0
        detected_anomalies = []
        for anomaly_id in range(len(self._anomalies)):
            anomaly = self._anomalies[anomaly_id]
            ambiguous = self._ambiguous_inst[anomaly_id]

            max_score = self._sum_of_func(anomaly.get_time()[0], anomaly.get_time()[1],
                                          anomaly.get_time()[0], anomaly.get_time()[1], self._uniform_func)

            score = 0.0
            for prediction in self._predictions:
                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)

            if min(1.0, score / max_score) > self._theta:
                total_score += 1.0
                detected_anomalies.append(anomaly)

        if len(self._anomalies) == 0:
            return 0.0, []
        else:
            return total_score / len(self._anomalies), detected_anomalies

    # return a value with the detected prediction lists
    def TaP_d(self) -> {float, list}:
        correct_predictions = []
        total_score = 0.0
        for prediction in self._predictions:
            max_score = prediction.get_time()[1] - prediction.get_time()[0] + 1

            score = 0.0
            for anomaly_id in range(len(self._anomalies)):
                anomaly = self._anomalies[anomaly_id]
                ambiguous = self._ambiguous_inst[anomaly_id]

                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)

            if (score/max_score) > self._theta:
                total_score += 1.0
                correct_predictions.append(prediction)

        if len(self._predictions) == 0:
            return 0.0, []
        else:
            return total_score / len(self._predictions), correct_predictions


    def _detect(self, src_range: Term, ranges: list, theta: int) -> bool:
        rest_len = src_range.get_time()[1] - src_range.get_time()[0] + 1
        for dst_range in ranges:
            len = self._overlapped_len(src_range, dst_range)
            if len != -1:
                rest_len -= len
        return (float)(rest_len) / (src_range.get_time()[1] - src_range.get_time()[0] + 1) <= (1.0 - theta)

    def _overlapped_len(self, range1: Term, range2: Term) -> int:
        detected_start = max(range1.get_time()[0], range2.get_time()[0])
        detected_end = min(range1.get_time()[1], range2.get_time()[1])

        if detected_end < detected_start:
            return 0
        else:
            return detected_end - detected_start + 1

    def _min_max_norm(self, value: int, org_min: int, org_max: int, new_min: int, new_max: int) -> float:
        return (float)(new_min) + (float)(value - org_min) * (new_max - new_min) / (org_max - org_min)

    def _decaying_func(self, val: float) -> float:
        assert (-6 <= val <= 6)
        return 1 / (1 + math.exp(val))

    def _ascending_func(self, val: float) -> float:
        assert (-6 <= val <= 6)
        return 1 / (1 + math.exp(val * -1))

    def _uniform_func(self, val: float) -> float:
        return 1.0

    def _sum_of_func(self, start_time: int, end_time: int, org_start: int, org_end: int,
                     func: Callable[[float], float]) -> float:
        val = 0.0
        for timestamp in range(start_time, end_time + 1):
            val += func(self._min_max_norm(timestamp, org_start, org_end, -6, 6))
        return val

    def _overlap_and_subsequent_score(self, anomaly: Term, ambiguous: Term, prediction: Term) -> float:
        score = 0.0

        detected_start = max(anomaly.get_time()[0], prediction.get_time()[0])
        detected_end = min(anomaly.get_time()[1], prediction.get_time()[1])

        score += self._sum_of_func(detected_start, detected_end,
                                   anomaly.get_time()[0], anomaly.get_time()[1], self._uniform_func)

        detected_start = max(ambiguous.get_time()[0], prediction.get_time()[0])
        detected_end = min(ambiguous.get_time()[1], prediction.get_time()[1])

        score += self._sum_of_func(detected_start, detected_end,
                                   ambiguous.get_time()[0], ambiguous.get_time()[1], self._decaying_func)

        return score

    def TaR_p(self) -> float:
        total_score = 0.0
        for anomaly_id in range(len(self._anomalies)):
            anomaly = self._anomalies[anomaly_id]
            ambiguous = self._ambiguous_inst[anomaly_id]

            max_score = self._sum_of_func(anomaly.get_time()[0], anomaly.get_time()[1],
                                          anomaly.get_time()[0], anomaly.get_time()[1], self._uniform_func)

            score = 0.0
            for prediction in self._predictions:
                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)

            total_score += min(1.0, score/max_score)

        if len(self._anomalies) == 0:
            return 0.0
        else:
            return total_score / len(self._anomalies)

    def TaP_p(self) -> float:
        total_score = 0.0
        for prediction in self._predictions:
            max_score = prediction.get_time()[1] - prediction.get_time()[0] + 1

            score = 0.0
            for anomaly_id in range(len(self._anomalies)):
                anomaly = self._anomalies[anomaly_id]
                ambiguous = self._ambiguous_inst[anomaly_id]

                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)

            total_score += score/max_score

        if len(self._predictions) == 0:
            return 0.0
        else:
            return total_score / len(self._predictions)