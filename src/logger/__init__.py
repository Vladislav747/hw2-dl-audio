from src.logger.cometml import CometMLWriter
from src.logger.logger import setup_logging
from src.logger.wandb import WandBWriter


def get_logger(name: str = "train"):
    return logging.getLogger(name)
