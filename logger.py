# https://stackoverflow.com/a/41477104/6274861

import sys

class Logger(object):
    def __init__(self, log_filepath):
        self.to_log = False
        self.terminal = sys.stdout
        self.log = open(log_filepath, "w")

    def __del__(self):
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        if self.to_log:
            self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        pass
    
    def start(self):
        self.to_log = True 
    
    def stop(self):
        self.to_log = False