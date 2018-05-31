import os
import sys
import logging


class LoggerWriter(object):
    def __init__(self, name, filename):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.datefmt = '%Y/%m/%d %H:%M:%S'
        self.set_default_log_handler()
        self.set_log_handler_to_file(filename)

        

    def set_default_log_handler(self):
        handler1 = logging.StreamHandler()
        handler1.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s",
            datefmt=self.datefmt)
            )
        self.logger.addHandler(handler1)

    def set_log_handler_to_file(self, filename):
        if os.path.exists(filename): 
            
            os.remove(filename)
        handler2 = logging.FileHandler(filename=filename) 
        handler2.setLevel(logging.DEBUG)    
        handler2.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s",
            datefmt=self.datefmt)
            )  
        self.logger.addHandler(handler2)

    def write(self, message):
        if message != "\n" or message != "":
            self.logger.info(message)

    def flush(self):
        handler = self.logger.handlers[0]
        handler.flush()

logger = sys.stderr
def set_logging(name, filename):
    global logger
    logger = LoggerWriter(name, filename + "." + name)
    
def trace(*args):
    global logger
    logger.write(" ".join(map(lambda x:str(x), args)))