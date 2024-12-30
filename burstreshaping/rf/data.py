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
        
        if tims[0] > 0.0:
            s_tim = tims[0]
            tims = [x-s_tim for x in tims]
            
        self.tims = tims
        self.seqs = seqs
        
        self.label = label

    def sum_burst_to_size_seqs(self, sum_burst, size):
        seqs = [size] * (sum_burst // size)
        if sum_burst % size > 0:
            seqs += [sum_burst % size]
        return seqs

    def get_time(self):
        return self.tims[-1]

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
        burst_time = []
        result_seqs = []
        result_tims = []
        for time, packet in zip(self.tims, self.seqs):
            if last_time is None:
                last_time = time
            if packet < 0 or packet == 536 or packet == 1050:
                
                if len(burst) > 0:
                    reshpae_burst = self.burst_reshaping(burst, ncpp)
                    result_seqs.extend(reshpae_burst)
                    if len(reshpae_burst)>1:
                        inter_time = (burst_time[-1] - burst_time[0]) / (len(reshpae_burst)-1)
                        result_tims.extend([burst_time[0]+i*inter_time for i in range(len(reshpae_burst))])
                    else:
                        result_tims.extend([burst_time[0]+(burst_time[-1] - burst_time[0])/2])
                    
                    
                    burst = []
                    burst_time = []
                result_seqs.append(packet)
                result_tims.append(time)
            elif packet > 0:
                if len(burst) > 0 and (time - last_time) > inthr:
                    reshpae_burst = self.burst_reshaping(burst, ncpp)
                    result_seqs.extend(reshpae_burst)
                    if len(reshpae_burst)>1:
                        inter_time = (burst_time[-1] - burst_time[0]) / (len(reshpae_burst)-1)
                        result_tims.extend([burst_time[0]+i*inter_time for i in range(len(reshpae_burst))])
                    else:
                        result_tims.extend([burst_time[0]+(burst_time[-1] - burst_time[0])/2])
                    
                    burst = []
                    burst_time = []
                burst.append(packet)
                burst_time.append(time)
            last_time = time
            
        if len(burst) > 0:
            reshpae_burst = self.burst_reshaping(burst, ncpp)
            result_seqs.extend(reshpae_burst)
            if len(reshpae_burst)>1:
                inter_time = (burst_time[-1] - burst_time[0]) / (len(reshpae_burst)-1)
                result_tims.extend([burst_time[0]+i*inter_time for i in range(len(reshpae_burst))])
            else:
                result_tims.extend([burst_time[0]+(burst_time[-1] - burst_time[0])/2])
                    
        return result_tims, result_seqs

    def get_rf_input(self, seqs, tims, length, time_slot):
        # print(tims[-1], length, time_slot, seqs[-1] - (length-1) * time_slot)
        feature = [[0 for _ in range(length)], [0 for _ in range(length)]]
        # start_time = tims[-1] - (length-1) * time_slot
        for i in range(0, len(seqs)):
            size = abs(seqs[i])
            if seqs[i] > 0:
                # if tims[i] <= start_time:
                #     feature[0][0] += size
                # else:
                    idx = int((tims[i]) / time_slot)
                    if idx<length:
                    # print(tims[i], start_time, idx)
                        feature[0][idx] += size
                    else:
                        feature[0][-1] += size
            if seqs[i] < 0:
                # if tims[i] <= start_time:
                #     feature[1][0] += size
                # else:
                    idx = int((tims[i]) / time_slot)
                    if idx<length:
                    # print(tims[i], start_time, idx)
                        feature[1][idx] += size
                    else:
                        feature[0][-1] += size
        # print(feature[0][:100],feature[1][:100])
        return feature

class RFDataset:
    def __init__(
        self, data_path, inthr, ncpp, length, train_rate, nb_class, max_matrix_len, time_slot, type_list=[], 
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
        
        self.max_matrix_len = max_matrix_len
        self.time_slot = time_slot
        
        self.train_traces, self.val_traces = self.load_dataset()
        
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
        seqs = []
        label = []
        for trace in traces:
            t_seqs = trace.seqs
            t_tims = trace.tims
            # seqs, tims, length, max_time
            if 'raw_seqs' in self.type_list:
                represent_seq = trace.get_rf_input(seqs = t_seqs, tims = t_tims, length = self.max_matrix_len, time_slot = self.time_slot)
                seqs.append(represent_seq)
                label.append(trace.label)
                
            if 'burst_reshaping' in self.type_list:
                t_tims, t_seqs = trace.interval_defined_burst_reshaping(inthr=self.inthr, ncpp=self.ncpp)
                represent_seq = trace.get_rf_input(seqs = t_seqs, tims = t_tims, length = self.max_matrix_len, time_slot = self.time_slot)
                seqs.append(represent_seq)
                label.append(trace.label)
            
            
        data = np.array(seqs).astype("float32")
        label = np.array(label).astype("float32")
        data = data[:, :, :, np.newaxis]
        label = np_utils.to_categorical(label, self.nb_classes)
        return data, label
            

    def pad_seqs(self, seqs, length):
        if len(seqs)>length:
            return seqs[:length]
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
