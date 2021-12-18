import logging
import os
import glob
import re

class logger:
    LATEST = 'LATEST.log'
    LOG_DIR = 'logs'

    def __init__(self):
        # create logger
        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(logging.INFO)

        # create file handler which logs messages
        self.move_old()
        file_handler = logging.FileHandler(f'{self.LOG_DIR}/{self.LATEST}')
        file_handler.setLevel(logging.DEBUG)

        # create a console handler to print to screen
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log(self, msg):
        self.logger.log(logging.INFO, msg)
    
    def err(self, msg):
        self.logger.log(logging.ERROR, msg)
    
    def warn(self, msg):
        self.logger.log(logging.WARN, msg)
    
    def move_old(self):
        if os.path.isdir(self.LOG_DIR):
            os.chdir(self.LOG_DIR)
            if os.path.isfile(self.LATEST):
                files = sorted(glob.glob('old*.log'), reverse=True)
                old_filename = 'old1.log'
                if len(files) > 0:
                    latest = files[0]
                    pattern = re.compile('[0-9]')
                    match = pattern.search(latest)
                    if match is not None:
                        iter = int(match.group())
                        old_filename = f'old-{iter + 1}.log'
                with open(self.LATEST, 'r') as latest:
                    with open(old_filename, 'w') as old:
                        for line in latest:
                            old.write(f'{line}')
                os.remove(self.LATEST)
            os.chdir('..')
        else:
            os.mkdir(self.LOG_DIR)