import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras import backend as K
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
    def on_epoch_end(self, epoch, logs=None):
        # 记录每个epoch的结束
        # print(logs)
        logger.info(
            f"Epoch {epoch + 1} ended - loss: {logs.get('loss'):.4f} , accuracy: {logs.get('accuracy'):.4f} , val_loss: {logs.get('val_loss'):.4f} , val_accuracy: {logs.get('val_accuracy'):.4f}"
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

with open(log_file,'w') as file:
    file.truncate()
    
logger.add(
    log_file,
    rotation="500MB",
    encoding="utf-8",
    enqueue=True,
    retention="10 days",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {file.name} - {function} : {message}",
)

def train_Pseudolabel_DF(conf, confidence_threshold):
    logger.info(f"new training : {confidence_threshold = }")

    model_conf = conf["model_conf"]
    model = DFNet().build(
        input_shape=(model_conf["length"], 1), classes=model_conf["nb_class"]
    )

    opt = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

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

    train_data, train_label = proxyset_generator.get_training_data(trace_type="train",seq_type="burst_reshaping")
    

    unlabeled_train_data = hsset_generator.train_data
    
    logger.info("initial training for 10 epochs...")
    model.fit(
        train_data,
        train_label,
        batch_size=model_conf["batch_size"],
        epochs = 10,
        validation_data=(hsset_generator.val_data, hsset_generator.val_label),
        verbose=0,
        callbacks=[loguru_callback],
    )


    for _ in range(model_conf["nb_epoch"] // model_conf["pseudo_per_epoch"]):
        
        train_data_predictions = model.predict(proxyset_generator.train_data)
        train_confidence_index = np.where(train_data_predictions.max(axis=1) < confidence_threshold)
        
        predictions = model.predict(unlabeled_train_data)
        confidence_index = np.where(predictions.max(axis=1) > confidence_threshold)
        
        train_data = np.concatenate(
            [proxyset_generator.train_data[train_confidence_index], unlabeled_train_data[confidence_index]]
        )
        train_label = np.concatenate(
            [
                proxyset_generator.train_label[train_confidence_index],
                np_utils.to_categorical(
                    np.argmax(predictions[confidence_index], -1),
                    model_conf["nb_class"],
                ),
                # predictions[confidence_index],
            ]
        )
        
        logger.info(
            f"pseudo_steps: {_+1}/{(model_conf['nb_epoch'] // model_conf['pseudo_per_epoch'])}, train_data shape is {train_data.shape}."
        )
        
        model.fit(
            train_data,
            train_label,
            batch_size=model_conf["batch_size"],
            epochs = model_conf["pseudo_per_epoch"],
            validation_data=(hsset_generator.val_data, hsset_generator.val_label),
            verbose=0,
            callbacks=[loguru_callback],
        )

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
    generator = PLLDataset(**hs_data_conf)

    predictions = model.predict(generator.val_data)

    accuracy, precision, recall, f1 = find_accuracy(
        predictions=predictions, actual=generator.val_label, show_prdiction_dict=False
    )

    logger.info(
        f"Testing result: {accuracy=:.4f}, {precision=:.4f}, {recall=:.4f}, {f1=:.4f}"
    )


if __name__ == "__main__":
    conf = read_yaml("./config.yaml")
    logger.info(conf)
    model = train_Pseudolabel_DF(conf, confidence_threshold=0.95)
    # model = train_DF(conf)
    
    test(conf, model)
    # model_conf = conf["model_conf"]
    # test(conf, None, model_path=Path(model_conf["save_path"])
    #         / f"{model_conf['save_name_head']}_epoch{model_conf['nb_epoch']}.h5")
