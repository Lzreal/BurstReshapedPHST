import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from loguru import logger
from data import PLLDataset, find_accuracy
from pathlib import Path
from keras.optimizers import Adamax
from keras.callbacks import Callback
from keras.models import load_model
from DFNet import DFNet
from keras.utils import np_utils
import numpy as np
from datetime import date
import yaml
import random
import tensorflow as tf

class LoguruCallback(Callback):
    def __init__(self):
        self.total_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        # 记录每个epoch的结束
        # print(logs)
        self.total_epoch += 1
        logger.info(
            f"Epoch {self.total_epoch} ended - loss: {logs.get('loss'):.4f} , accuracy: {logs.get('accuracy'):.4f} , val_loss: {logs.get('val_loss'):.4f} , val_accuracy: {logs.get('val_accuracy'):.4f}"
        )


def read_yaml(file_path):
    with open(file_path, "r") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(e)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# 获取今日日期
today = date.today()

# 转换为字符串格式
today_str = today.strftime("%Y-%m-%d")

log_file = "./experiments.log"

# with open(log_file,'w') as file:
#     file.truncate()
    
logger.add(
    log_file,
    rotation="500MB",
    encoding="utf-8",
    enqueue=True,
    retention="10 days",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {file.name} - {function} : {message}",
)

def train_Pseudolabel_DF(conf,desc,params):
    
    model_conf = conf["model_conf"]
    confidence_threshold = model_conf["confidence_threshold"]
    
    logger.info(f"new training : {desc}={params}")
    logger.info(conf)
    
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
    proxyset_generator = PLLDataset(**proxy_data_conf)

    hs_data_conf = conf["data_conf"]["train_conf"]["hs_conf"]
    hs_data_conf["length"] = model_conf["length"]
    hs_data_conf["nb_class"] = model_conf["nb_class"]
    hsset_generator = PLLDataset(**hs_data_conf)

    logger.info("Start training....")

    train_data, train_label = (
        proxyset_generator.train_data,
        proxyset_generator.train_label,
    )

    unlabeled_train_data = hsset_generator.train_data
    
    logger.info(f"initial training for {model_conf['init_epoch']} epochs...")
    model.fit(
        train_data,
        train_label,
        batch_size=model_conf["batch_size"],
        epochs = model_conf['init_epoch'],
        validation_data=(hsset_generator.val_data, hsset_generator.val_label),
        verbose=0,
        callbacks=[loguru_callback],
    )
    
    for _ in range(model_conf["nb_epoch"] // model_conf["pseudo_per_epoch"]):
        
        
        predictions = model.predict(unlabeled_train_data)
        confidence_index = np.where(predictions.max(axis=1) > confidence_threshold)
        
        train_data = np.concatenate(
            [proxyset_generator.train_data, unlabeled_train_data[confidence_index]]
        )
        train_label = np.concatenate(
            [
                proxyset_generator.train_label,
                np_utils.to_categorical(
                    np.argmax(predictions[confidence_index], -1),
                    model_conf["nb_class"],
                ),
            ]
        )
        
        logger.info(
            f"pseudo_steps: {_+1}/{(model_conf['nb_epoch'] // model_conf['pseudo_per_epoch'])}, train_data shape is {train_data.shape}."
        )
        
        model.fit(
            train_data,
            train_label,
            batch_size = model_conf["batch_size"],
            epochs = model_conf["pseudo_per_epoch"],
            validation_data = (hsset_generator.val_data, hsset_generator.val_label),
            verbose = 0,
            callbacks = [loguru_callback],
        )

    if not Path(model_conf["save_path"]).exists():
        Path(model_conf["save_path"]).mkdir(parents=True)

    if model_conf["save_model"]:
        # save model
        try:
            save_path = (
                Path(model_conf["save_path"])
                / f"{model_conf['save_name_head']}_{desc}-{params}.h5"
            )
            model.save(save_path)
            logger.info(f"Save model to {save_path}.")
        except Exception as e:
            logger.info(f"save model failed. {e}")

    return model


def test(conf, model=None, model_path=None,desc=None, params=None):
    logger.info("test data...")
    if model is None and model_path is not None:
        model = load_model(model_path)

    model_conf = conf["model_conf"]
    hs_data_conf = conf["data_conf"]["test_conf"]
    hs_data_conf["length"] = model_conf["length"]
    hs_data_conf["nb_class"] = model_conf["nb_class"]
    generator = PLLDataset(**hs_data_conf)

    predictions = model.predict(generator.val_data)

    accuracy, precision, recall, f1 = find_accuracy(
        predictions=predictions, actual=generator.val_label, show_prdiction_dict=False
    )

    logger.info(
        f"Testing result: params {desc}={params} {accuracy=:.4f}, {precision=:.4f}, {recall=:.4f}, {f1=:.4f}"
    )
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    
    # for confidence_threshold in [0.9,0.91,0.92,0.95,0.98,0.99]:
    #     conf = read_yaml("./config.yaml")
    #     tmp_config = conf
    #     tmp_config['model_conf']['confidence_threshold'] = confidence_threshold
    #     model = train_Pseudolabel_DF(tmp_config,desc="confidence_threshold",params=confidence_threshold)
    #     test(conf, model,desc="confidence_threshold",params=confidence_threshold)
    
    # for pseudo_times in [5,6,7,8,9]:
    #     conf = read_yaml("./config.yaml")
    #     tmp_config = conf
    #     tmp_config['model_conf']['nb_epoch'] = pseudo_times * tmp_config['model_conf']['pseudo_per_epoch']
    #     model = train_Pseudolabel_DF(tmp_config,desc="pseudo_times",params=pseudo_times)
    #     test(conf, model, desc="pseudo_times",params=pseudo_times)
    
    # for pseudo_per_epoch in [5,10,15,20]:
    #     conf = read_yaml("./config.yaml")
    #     tmp_config = conf
    #     tmp_config['model_conf']['pseudo_per_epoch'] = pseudo_per_epoch
    #     model = train_Pseudolabel_DF(tmp_config,desc="pseudo_per_epoch",params=pseudo_per_epoch)
    #     test(conf, model, desc="pseudo_per_epoch",params=pseudo_per_epoch)
        
    # for init_epoch in [5,10,15,20,25]:
    #     conf = read_yaml("./config.yaml")
    #     tmp_config = conf
    #     tmp_config['model_conf']['init_epoch'] = init_epoch
    #     model = train_Pseudolabel_DF(tmp_config,desc="init_epoch",params=init_epoch)
    #     test(conf, model ,desc="init_epoch",params=init_epoch)
    
    # result_list = [[],[],[],[]]
    # for _ in range(10):
    conf = read_yaml("./config.yaml")
    model = train_Pseudolabel_DF(conf, desc="total",params=1)
    result = test(conf, model, desc="total",params=1)
    # for index in range(4):
    #     result_list[index].append(result[index])
            
    # logger.info(f'total result list: {result_list}')        
    # logger.info(f'total result: mean accuracy is {np.mean(result_list[0]):.4f} , mean precision is {np.mean(result_list[1]):.4f}, mean recall is {np.mean(result_list[2]):.4f}, mean f1 is {np.mean(result_list[3]):.4f}. ')
    # model_conf = conf["model_conf"]
    # test(conf, None, model_path=Path(model_conf["save_path"])
    #         / f"{model_conf['save_name_head']}_epoch{model_conf['nb_epoch']}.h5")
