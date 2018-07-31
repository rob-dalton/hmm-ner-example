import logging

def initialize_logging():
    logging.basicConfig(
        filename="./output.log",
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )
