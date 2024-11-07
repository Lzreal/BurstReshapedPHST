import numpy as np
from pathlib import Path
import random
import json
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from pprint import pprint
from loguru import logger
import yaml
import json

class Trace:
    def __init__(self, tims=None, seqs=None, label=None):
        if tims[0] == 0.0:
            self.tims = tims
        else:
            self.tims = [x - tims[0] for x in tims]
            
        self.seqs = seqs
        self.label = label

    def sum_burst_to_size_seqs(self, sum_burst, size):
        seqs = [size] * (sum_burst // size)
        if sum_burst % size > 0:
            seqs += [sum_burst % size]
        return seqs

    def get_var_cnn_metadata(self, seqs, tims, length):
        dir_seq = np.zeros(length, dtype=np.int64)
        time_seq = np.zeros(length, dtype=np.float32)
        total_time = float(seqs[-1])
        total_incoming = 0
        total_outgoing = 0
        
        for packet_num in range(len(seqs)):
            curr_time = tims[packet_num]
            # curr_dir = np.sign(seqs[packet_num])
            curr_dir = seqs[packet_num]
            if packet_num < length:
                # print(curr_dir)
                dir_seq[packet_num] = curr_dir
                time_seq[packet_num] = curr_time
            
            if curr_dir > 0:
                total_outgoing += 1
            elif curr_dir < 0:
                total_incoming += 1
        # print(seqs,dir_seq)
        total_packets = total_incoming + total_outgoing
        if total_packets == 0:
            metadata = np.zeros(7, dtype=np.float32)
        else:
            metadata = np.array([total_packets, total_incoming, total_outgoing,
                                total_incoming / total_packets,
                                total_outgoing / total_packets,
                                total_time, total_time / total_packets],
                                dtype=np.float32)
        # print(seqs[:10],dir_seq[:10])
        return dir_seq, time_seq, metadata
        
        
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

class VarCNNDataset:
    def __init__(
        self, data_path, inthr, ncpp, length, train_rate, nb_class, mixture, is_val = False, type_list=[]
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
        self.mixture = mixture
        # print(len(self.train_traces),len(self.val_traces))
        if len(self.train_traces) > 0:
            self.train_data, self.train_label = self.get_total_seqs(self.train_traces)
            logger.info(
                f"Training Set : data = {self.train_data[0].shape}, {self.train_data[1].shape}, label = {self.train_label.shape}"
            )
        if len(self.val_traces) > 0:
            if is_val:
                logger.info('val set is validation_data, mixture is burst_reshaping only!')
                self.val_data, self.val_label = self.get_total_seqs(self.val_traces, type_list = ['burst_reshaping'])
            else:
                self.val_data, self.val_label = self.get_total_seqs(self.val_traces)
            logger.info(
                f"Val/Test Set : data = {self.val_data[0].shape}, {self.val_data[1].shape}, label = {self.val_label.shape}"
            )

    def load_dataset(self):
        train_trace_list = []
        val_trace_list = []
        with open(self.dataset_path, "r") as file:
            json_data = json.load(file)
        for key in json_data.keys():
            total_size = len(json_data[key])
            index_list = list(range(total_size))
            random.shuffle(index_list)
            for index in index_list[: int(total_size * self.split_rate)]:
                _ = json_data[key][index]
                train_trace_list.append(Trace(seqs=_["s"], tims=_["t"], label=int(key)))
            
            for index in index_list[int(total_size * self.split_rate) :]:
                _ = json_data[key][index]
                val_trace_list.append(Trace(seqs=_["s"], tims=_["t"], label=int(key)))
            
        return train_trace_list, val_trace_list
    
    def get_total_seqs(self, traces, type_list = None):
        seqs = []
        if 'metadata' in self.mixture:
            mds = []
        label = []
        
        mixture = self.mixture
        
        if type_list is None:
            type_list = self.type_list
            
        for trace in traces:
            t_seqs = trace.seqs
            t_tims = trace.tims
            
            if 'raw_seqs' in type_list:
                dir_seq, time_seq, metadata = trace.get_var_cnn_metadata(seqs = t_seqs, tims = t_tims, length = self.length)
                
                if 'dir' in mixture:
                    seqs.append(dir_seq)
                elif 'time' in mixture:
                    seqs.append(time_seq)
                if 'metadata' in self.mixture:
                    mds.append(metadata)
                label.append(trace.label)
            
            if 'burst_reshaping' in type_list:
                t_tims, t_seqs = trace.interval_defined_burst_reshaping(inthr=self.inthr, ncpp=self.ncpp)
                dir_seq, time_seq, metadata = trace.get_var_cnn_metadata(seqs = t_seqs, tims = t_tims, length = self.length)
                if 'dir' in mixture:
                    seqs.append(dir_seq)
                elif 'time' in mixture:
                    seqs.append(time_seq)
                if 'metadata' in self.mixture:    
                    mds.append(metadata)
                label.append(trace.label)

        data = np.array(seqs).astype("float32")
        
        label = np.array(label).astype("float32")
        # print(data.shape,mds.shape,label.shape)
        data = data[:, :, np.newaxis]
        label = np_utils.to_categorical(label, self.nb_classes)
        if 'metadata' in self.mixture:
            mds = np.array(mds).astype("float32")
            return [data,mds], label
        return data,label

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

def main():
    pass

def read_yaml(file_path):
    with open(file_path, "r") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(e)

if __name__ == '__main__':
    
    conf = read_yaml("./config.yaml")
    
    with open('config.json') as config_file:
        config = json.load(config_file)
    
    config.update(conf)    
    logger.info(config)    
    
    mixture = config['mixture']
    
    for mixture_num, inner_comb in enumerate(mixture): 
        model_conf = conf["model_conf"]
        proxy_data_conf = conf["data_conf"]["train_conf"]["proxy_conf"]
        proxy_data_conf["length"] = model_conf["length"]
        proxy_data_conf["nb_class"] = model_conf["nb_class"]
        proxy_data_conf["mixture"] = mixture
        proxy_dataset = VarCNNDataset(**proxy_data_conf, is_val = True)
        
    pass
