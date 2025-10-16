import logging
import sys

try:
    import colorlog
except ImportError:  # pragma: no cover - fallback when optional dependency missing
    colorlog = None


def create_logger(level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(level)
    log_config = {
        "DEBUG": {"level": 10, "color": "purple"},
        "INFO": {"level": 20, "color": "green"},
        "WARNING": {"level": 30, "color": "yellow"},
        "ERROR": {"level": 40, "color": "red"},
    }
    if colorlog:
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s: %(message)s",
            log_colors={key: conf["color"] for key, conf in log_config.items()},
        )
    else:
        formatter = logging.Formatter("[%(asctime)-15s] [%(levelname)8s]: %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    return logger


logger = create_logger()
