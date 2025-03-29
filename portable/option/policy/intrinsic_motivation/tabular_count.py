import os 
import gin
import pickle
from collections import defaultdict
import numpy as np
import logging
logger = logging.getLogger(__name__)

@gin.configurable
class TabularCount():
    def __init__(self,
                 beta) -> None:
        self.beta = beta 
        self.counts = defaultdict(int)
        
        logger.info("Tabular Count hps")
        logger.info("======================================")
        logger.info("======================================")
        logger.info("Beta: {}".format(beta))
        logger.info("======================================")
        logger.info("======================================")
    
    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        save_file = os.path.join(dir, 'counts.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(self.counts, f)
    
    def load(self, dir):
        save_file = os.path.join(dir, 'counts.pkl')
        with open(save_file, "rb") as f:
            self.counts = pickle.load(f)
    
    def get_bonus(self, state):
        self.counts[state] += 1
        return self.beta/np.sqrt(self.counts[state])


