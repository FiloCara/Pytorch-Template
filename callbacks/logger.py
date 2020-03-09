import logging

class Logger:

    def __init__(self, filepath):
        
        self.filepath = filepath

        if not self.filepath:

            self.filepath = "./training.log"

        logging.basicConfig(filename=self.filepath, level=logging.INFO)
        logging.info("This is the logger")
        pass

    def __call__(self):
        pass
