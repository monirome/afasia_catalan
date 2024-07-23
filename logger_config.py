# import logging
# from colorlog import ColoredFormatter


# def setup_logger():
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.DEBUG)
#     logger.propagate = False
#
#     stream_handler = logging.StreamHandler()
#     formatter = ColoredFormatter(
#         "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
#         datefmt=None,
#         reset=True,
#         log_colors={
#             'DEBUG': 'cyan',
#             'INFO': 'green',
#             'WARNING': 'yellow',
#             'ERROR': 'red',
#             'CRITICAL': 'red,bg_white',
#         },
#         secondary_log_colors={},
#         style='%'
#     )
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)
#
#     return logger
import logging

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
