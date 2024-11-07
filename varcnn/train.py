import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras import backend as K
from loguru import logger
from data import find_accuracy, VarCNNDataset
from pathlib import Path
from keras.optimizers import Adamax
from keras.callbacks import Callback
from keras.models import load_model
from model import get_model
from keras.utils import np_utils
import numpy as np
from datetime import date
import yaml
import random
import tensorflow as tf
import sys
import json

config_path = sys.argv[1]

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class LoguruCallback(Callback):
    def __init__(self):
        self.total_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        # 记录每个epoch的结束
        # print(logs)
        logger.info(
            f"Epoch {self.total_epoch + 1} ended - loss: {logs.get('loss'):.4f} , accuracy: {logs.get('accuracy'):.4f}"
        )
        self.total_epoch+=1


def read_yaml(file_path):
    with open(file_path, "r") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(e)

# 获取今日日期
today = date.today()

# 转换为字符串格式
today_str = today.strftime("%Y-%m-%d")

logger.add(
    f"./log/{today_str}.log",
    rotation="500MB",
    encoding="utf-8",
    enqueue=True,
    retention="10 days",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {file.name} - {function} : {message}",
)


def train_DF(conf):
    logger.info("new training : ----")

    model_conf = conf["model_conf"]
    model = DFNet().build(
        input_shape=(model_conf["length"], 1), classes=model_conf["nb_class"]
    )
    opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    loguru_callback = LoguruCallback()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    logger.info("Model compiled.")
    proxy_data_conf = conf["data_conf"]["train_conf"]["proxy_conf"]
    proxy_data_conf["length"] = model_conf["length"]
    proxy_data_conf["nb_class"] = model_conf["nb_class"]
    dataset = Dataset(**proxy_data_conf)
    logger.info("Start training....")

    model.fit(
        dataset.train_data,
        dataset.train_label,
        batch_size=model_conf["batch_size"],
        epochs=model_conf["nb_epoch"],
        verbose=0,
        callbacks=[loguru_callback],
    )

    # generator.next_train().__next__()

    if not Path(model_conf["save_path"]).exists():
        Path(model_conf["save_path"]).mkdir(parents=True)

    # save model
    try:
        save_path = (
            Path(model_conf["save_path"])
            / f"{model_conf['save_name_head']}_epoch{model_conf['nb_epoch']}.h5"
        )
        model.save(save_path)
        logger.info(f"Save model to {save_path}.")
    except Exception as e:
        logger.info(f"save model failed. {e}")
    return model


def test(conf, model=None, model_path=None):
    logger.info("test data...")
    if model is None and model_path is not None:
        model = load_model(model_path)

    model_conf = conf["model_conf"]
    hs_data_conf = conf["data_conf"]["test_conf"]
    hs_data_conf["length"] = model_conf["length"]
    hs_data_conf["nb_class"] = model_conf["nb_class"]
    generator = Dataset(**hs_data_conf)

    predictions = model.predict(generator.val_data)

    accuracy, precision, recall, f1 = find_accuracy(
        predictions=predictions, actual=generator.val_label, show_prdiction_dict=True
    )

    logger.info(
        f"Testing result: {accuracy=:.4f}, {precision=:.4f}, {recall=:.4f}, {f1=:.4f}"
    )

def train_predict_model(conf, model, mixture, callbacks):
    model_conf = conf["model_conf"]
    proxy_data_conf = conf["data_conf"]["train_conf"]["hs_conf"]
    proxy_data_conf["length"] = model_conf["length"]
    proxy_data_conf["nb_class"] = model_conf["nb_class"]
    proxy_data_conf["mixture"] = mixture
    proxy_dataset = VarCNNDataset(**proxy_data_conf,is_val=True)
    

    hs_data_conf = conf["data_conf"]["test_conf"]
    hs_data_conf["length"] = model_conf["length"]
    hs_data_conf["nb_class"] = model_conf["nb_class"]
    hs_data_conf["mixture"] = mixture
    test_dataset = VarCNNDataset(**hs_data_conf)
    
    # hs_data_conf = conf["data_conf"]["train_conf"]["hs_conf"]
    # hs_data_conf["length"] = model_conf["length"]
    # hs_data_conf["nb_class"] = model_conf["nb_class"]
    # hs_dataset = VarCNNDataset(**hs_data_conf)
    
    print(mixture)
    
    # val_data,val_label = proxy_dataset.get_total_seqs(traces = proxy_dataset.val_traces, mixture = mixture)
    
    
    model.fit(proxy_dataset.train_data,proxy_dataset.train_label,epochs = conf['var_cnn_max_epochs'], verbose=2, callbacks=callbacks, validation_data=(proxy_dataset.val_data ,proxy_dataset.val_label ))
    # model.fit(train_data,train_label,epochs = conf['var_cnn_max_epochs'],verbose=1, callbacks=callbacks,validation_data = (val_data,val_label))
    
    
    model.load_weights('model_weights.h5')
    
    model.save("varcnn-{}.h5".format("dir" if 'dir' in mixture else "time"))

    predictions = model.predict(test_dataset.val_data)
    print(predictions.shape)
    acc,pre,recall,f1 = find_accuracy(predictions=predictions,actual=test_dataset.val_label)
    print(f'result: {acc = :.4f}, {pre = :.4f}, {recall = :.4f}, {f1 = :.4f},')
    np.savez_compressed("predictions-varcnn-{}.npz".format("dir" if 'dir' in mixture else "time"), predictions = predictions, labels = test_dataset.val_label)
    
def test_model2():
    dir_predictions, dir_labels = np.load("predictions-varcnn-dir.npz")['predictions'], np.load("predictions-varcnn-dir.npz")['labels']
    tim_predictions, tim_labels = np.load("predictions-varcnn-time.npz")['predictions'], np.load("predictions-varcnn-time.npz")['labels']
    
    total_predictions = dir_predictions + tim_predictions
    total_predictions/=2
    
    for name, predictions in zip(['dir','tim','total'],[dir_predictions,tim_predictions,total_predictions]):
        acc,pre,recall,f1 = find_accuracy(predictions=predictions,actual=dir_labels)
        logger.info(f'Result for {name}: {acc = :.4f}, {pre = :.4f}, {recall = :.4f}, {f1 = :.4f},')
    
    
    # model_conf = conf["model_conf"]
    # model.load_weights("model_weights.h5")
    # hs_data_conf = conf["data_conf"]["test_conf"]
    # hs_data_conf["length"] = model_conf["length"]
    # hs_data_conf["nb_class"] = model_conf["nb_class"]
    # hs_data_conf["mixture"] = mixture
    # test_dataset = VarCNNDataset(**hs_data_conf)
    
    # predictions = model.predict(test_dataset.val_data)
    # acc,pre,recall,f1 = find_accuracy(predictions=predictions,actual=test_dataset.val_label)
    # print(f'result: {acc = :.4f}, {pre = :.4f}, {recall = :.4f}, {f1 = :.4f},')

if __name__ == "__main__":
    conf = read_yaml("./config.yaml")
    
    with open('config.json') as config_file:
        config = json.load(config_file)
    
    config.update(conf)    
    logger.info(config)    
    
    mixture = config['mixture']
    for mixture_num, inner_comb in enumerate(mixture):
        model, callbacks = get_model(config, mixture_num, 103)
        # test_model2(model,config,inner_comb)
        sub_model_name = '_'.join(inner_comb)
        # print(model.summary())
        train_predict_model(config,model,inner_comb,callbacks)
        
    test_model2()    
        
    # conf = read_yaml(config_path)
    
    # # model = train_Pseudolabel_DF(conf, confidence_threshold=0.99)
    
    # model = train_DF(conf)
    # test(conf, model)
    
    # model_conf = conf["model_conf"]
    # test(conf, None, model_path = Path(model_conf["save_path"])
    #         / f"{model_conf['save_name_head']}_epoch{model_conf['nb_epoch']}.h5")
