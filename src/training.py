import os
from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model
from src.utils.model import save_model
from src.utils.model import save_plot
from src.utils.callbacks import get_callbacks

import pandas as pd
import argparse
import logging

# Logger
from src.utils.logger import setup_applevel_logger

# General logs
def loge(config_path):
    logs_name = "general_logs.log"
    config = read_config(config_path)
    log = setup_applevel_logger(config, file_name=logs_name)
    return log


# training
def training(config_path):

    config = read_config(config_path)
    #logging.info(config)
    # data management
    logging.info(">>> Data Loading>>>>")
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    #print(X_train.shape)
    logging.info(">>> Data loading completed >>>>")

    # create model
    logging.info(">>> Model created >>>>")
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)
    logging.info(">>> Model creation completed >>>>")

    # Training

    EPOCHS = config["params"]["epochs"]
    VALIDATION = (X_valid, y_valid)

    # create callbacks
    CALLBACK_LIST = get_callbacks(config, X_train)

    try:
        logging.info(">>>>> starting training >>>>>")
        history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION, callbacks=CALLBACK_LIST)
        logging.info("<<<<< training done successfully<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
    # save the model
    logging.info(">>> Saving models >>>>")
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    model_name=config["artifacts"]["model_name"]
    save_model(model, model_name , model_dir= model_dir_path )

    logging.info(f">>> Model saved Location: {model_dir_path}>>>>")

    #save the plot
    logging.info(">>> Saving plots >>>>")
    plots_dir = config["artifacts"]["plots_dir"]
    plots_dir_path= os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plots_dir_path, exist_ok=True)

    plots_name=config["artifacts"]["plots_name"]
    df= pd.DataFrame(history.history)
    save_plot(df,plots_name,plots_dir_path)
    logging.info(f">>> Plot saving done at {plots_dir_path} >>>>\n\n")





    




if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    loge(config_path=parsed_args.config)

    training(config_path=parsed_args.config)