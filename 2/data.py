import numpy as np
from pathlib import Path
import random
import json
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from pprint import pprint
from loguru import logger

class Trace:
    def __init__(self, tims=None, seqs=None, label=None):
        self.tims = tims
        self.seqs = seqs
        self.label = label

    def sum_burst_to_size_seqs(self, sum_burst, size):
        seqs = [size] * (sum_burst // size)
        if sum_burst % size > 0:
            seqs += [sum_burst % size]
        return seqs

    def burst_reshaping(self, burst, ncpp):
        sum_burst = sum(burst)
        # No need to reshape
        if sum_burst <= 1448:
            return [sum_burst]
        # TCP fragment
        if sum_burst <= 4070:
            return self.sum_burst_to_size_seqs(sum_burst, size=1448)

        # TCP Delay
        tls_seqs = self.sum_burst_to_size_seqs(sum_burst, size=4070)
        result = []
        tls_buff = tls_seqs[0]
        for tls in tls_seqs[1:]:
            if random.random() > ncpp:
                tls_buff += tls
            else:
                result.extend(self.sum_burst_to_size_seqs(tls_buff, 1448))
                tls_buff = tls
        if tls_buff > 0:
            result.extend(self.sum_burst_to_size_seqs(tls_buff, 1448))
        return result

    def interval_defined_burst_reshaping(self, inthr: float = 0.05, ncpp: float = 0.01):
        last_time: float = None
        burst = []
        result_seqs = []
        for time, packet in zip(self.tims, self.seqs):
            if last_time is None:
                last_time = time
            if packet < 0 or packet == 536 or packet == 1050:
                if len(burst) > 0:
                    result_seqs.extend(self.burst_reshaping(burst, ncpp))
                    burst = []
                result_seqs.append(packet)
            elif packet > 0:
                if len(burst) > 0 and (time - last_time) > inthr:
                    result_seqs.extend(self.burst_reshaping(burst, ncpp))
                    burst = []
                burst.append(packet)
            last_time = time
        if len(burst) > 0:
            result_seqs.extend(self.burst_reshaping(burst, ncpp))

        return result_seqs


class PLLDataset:
    def __init__(
        self, data_path, inthr, ncpp, length, train_rate, nb_class, type_list=[]
    ):
        data_path = Path(data_path)
        logger.info(f"----------------------")
        logger.info(
            f"Genrate Dataset : {data_path.name=}, {inthr=}, {ncpp=}, {length=}, {train_rate=}, {nb_class=}, {type_list=}"
        )
        self.dataset_path = data_path
        self.inthr = inthr
        self.ncpp = ncpp
        self.length = length
        self.nb_classes = nb_class
        self.split_rate = train_rate
        self.type_list = type_list
        self.train_traces, self.val_traces = self.load_dataset()
        
        # print(len(self.train_traces),len(self.val_traces))
        if len(self.train_traces) > 0:
            self.train_data, self.train_label = self.get_total_seqs(self.train_traces)
            logger.info(
                f"Training Set : data = {self.train_data.shape}, label = {self.train_label.shape}"
            )
        if len(self.val_traces) > 0:
            self.val_data, self.val_label = self.get_total_seqs(self.val_traces)
            logger.info(
                f"Val/Test Set : data = {self.val_data.shape}, label = {self.val_label.shape}"
            )

    def load_dataset(self):
        train_trace_list = []
        val_trace_list = []
        with open(self.dataset_path, "r") as file:
            json_data = json.load(file)
        for key in json_data.keys():
            total_size = len(json_data[key])
            for _ in json_data[key][: int(total_size * self.split_rate)]:
                train_trace_list.append(Trace(seqs=_["s"], tims=_["t"], label=int(key)))
            for _ in json_data[key][int(total_size * self.split_rate) :]:
                val_trace_list.append(Trace(seqs=_["s"], tims=_["t"], label=int(key)))
        return train_trace_list, val_trace_list

    def get_total_seqs(self, traces):
        data = []
        label = []
        for trace in traces:
            if "raw_seqs" in self.type_list:
                data.append(self.pad_seqs(trace.seqs, self.length))
                label.append(int(trace.label))
            if "burst_reshaping" in self.type_list:
                data.append(
                    self.pad_seqs(
                        trace.interval_defined_burst_reshaping(
                            inthr=self.inthr, ncpp=self.ncpp
                        ),
                        self.length,
                    )
                )
                label.append(int(trace.label))

            
        data = np.array(data).astype("float32")
        label = np.array(label).astype("float32")
        data = data[:, :, np.newaxis]
        label = np_utils.to_categorical(label, self.nb_classes)
        return data, label

    def pad_seqs(self, seqs, length):
        return np.pad(
            np.array(seqs),
            (0, length - len(seqs) + 1),
            mode="constant",
            constant_values=(0, 0),
        )[:length]


def find_accuracy(predictions, actual, show_prdiction_dict=False):
    predictions = np.argmax(predictions, axis=1)
    actual = np.argmax(actual, axis=1)

    if show_prdiction_dict:
        predictions_right_dic = {}
        for index in range(len(actual)):
            p = predictions[index]
            l = actual[index]
            if predictions_right_dic.get(l) is None:
                predictions_right_dic[l] = {"total": 0, "true": 0}
            predictions_right_dic[l]["total"] += 1
            if p == l:
                predictions_right_dic[l]["true"] += 1
        pprint(predictions_right_dic)

    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions, average="macro")
    recall = recall_score(actual, predictions, average="macro")
    f1 = f1_score(actual, predictions, average="macro")
    # print("accuracy_score:{}".format(accuracy))
    # print("precision_score:{}".format( precision))
    # print("recall_score:{}".format( recall))
    # print("f1_score:{}".format( f1))
    return accuracy, precision, recall, f1
