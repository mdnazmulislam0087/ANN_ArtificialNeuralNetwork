import logging

import os





def setup_applevel_logger(config, file_name=None):

    logs_dir = config["logs"]["logs_dir"]
    general_logs = config["logs"]["general_logs"]

    general_logs_dir_path = os.path.join(logs_dir, general_logs)
    os.makedirs(general_logs_dir_path, exist_ok=True)




    """format_log="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_fmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(format_log, date_fmt)"""

    logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"

    logger=logging.basicConfig(filename=os.path.join(general_logs_dir_path, file_name), level=logging.INFO, format=logging_str,
                        filemode="a")


    return logger


