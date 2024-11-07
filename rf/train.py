import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras import backend as K
from loguru import logger
from data import find_accuracy, RFDataset
from pathlib import Path
from keras.optimizers import Adamax,Adam
from keras.callbacks import Callback
from keras.models import load_model
from model import RF_Model
import numpy as np
from datetime import date
import yaml
import random
import tensorflow as tf
import time

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class LoguruCallback(Callback):
    def __init__(self):
        self.total_epoch = 0
        self.epoch_start_time = 0.0
        
    def on_epoch_begin(self, epoch, logs=None):    
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        # 记录每个epoch的结束
        # print(logs)
        text = f"Epoch {self.total_epoch + 1} ended, spend {(time.time()-self.epoch_start_time):.0f}s - loss: {logs.get('loss'):.4f} , accuracy: {logs.get('accuracy'):.4f}"
        if logs.get('val_accuracy') is not None:
            text += f" , val_loss: {logs.get('val_loss'):.4f} , val_accuracy: {logs.get('val_accuracy'):.4f}"
        logger.info(text)
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
    const_conf = conf["const_conf"]
    model_conf.update(const_conf)
    model = RF_Model().build(
        input_shape=(2, const_conf["max_matrix_len"], 1), classes=model_conf["nb_class"]
    )
    opt = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # opt = Adam(lr = 0.001, decay=0.001)
    loguru_callback = LoguruCallback()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    logger.info("Model compiled.")
    proxy_data_conf = conf["data_conf"]["train_conf"]["hs_conf"]
    proxy_data_conf.update(const_conf)
    dataset = RFDataset(**proxy_data_conf)
    logger.info("Start training....")

    const_conf = conf["const_conf"]
    hs_data_conf = conf["data_conf"]["test_conf"]
    hs_data_conf.update(const_conf)
    test_dataset = RFDataset(**hs_data_conf)
    
    
    model.fit(
        dataset.train_data,
        dataset.train_label,
        batch_size=model_conf["batch_size"],
        epochs=model_conf["nb_epoch"],
        verbose=0,
        callbacks=[loguru_callback],
        validation_data = (test_dataset.val_data,test_dataset.val_label),
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

    const_conf = conf["const_conf"]
    hs_data_conf = conf["data_conf"]["test_conf"]
    hs_data_conf.update(const_conf)
    generator = RFDataset(**hs_data_conf)

    predictions = model.predict(generator.val_data)

    accuracy, precision, recall, f1 = find_accuracy(
        predictions=predictions, actual=generator.val_label, show_prdiction_dict=False
    )

    logger.info(
        f"Testing result: {accuracy=:.4f}, {precision=:.4f}, {recall=:.4f}, {f1=:.4f}"
    )
    
if __name__ == "__main__":
    conf = read_yaml("./config.yaml") 
    
    model = train_DF(conf)
    test(conf,model)
    # test(conf, None, model_path = "saved_model/DF_epoch230.h5")