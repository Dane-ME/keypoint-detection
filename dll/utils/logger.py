import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", log_filename=None):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        if log_filename is None:
            log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
        
        log_path = os.path.join(log_dir, log_filename)
        
        self.logger = logging.getLogger("CustomLogger")
        self.logger.setLevel(logging.DEBUG)
        
        # Formatter
        log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        
        # File Handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(log_format)
        file_handler.setLevel(logging.DEBUG)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        console_handler.setLevel(logging.INFO)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)