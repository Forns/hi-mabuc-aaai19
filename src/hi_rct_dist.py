'''
  hi_rct_dist.py
  
  === Description ===
  Reward distribution class; provides basic storage and
  query capacity for discrete HI-RCT reward dists
  
  [!] Note: inefficient storage solution for HI-rewards,
  which is designed as such only for experimental purposes
  to be easily inspected
'''

import numpy as np
from hi_rct_utl import *

class HI_RCT_Dist:
    
    @staticmethod
    def average_dists (dist_list):
        result = HI_RCT_Dist(dist_list[0].ACTOR_COUNT, dist_list[0].X_DOM, dist_list[0].Y_DOM)
        for d in dist_list:
            result.dist += d.dist
        result.dist = np.floor(result.dist / len(dist_list))
        return result
    
    def __init__ (self, ACTOR_COUNT, X_DOM, Y_DOM, dist_copy=None):
        self.ACTOR_COUNT = ACTOR_COUNT
        self.X_DOM = X_DOM
        self.Y_DOM = Y_DOM
        X_CARD = len(X_DOM)
        Y_CARD = len(Y_DOM)
        intents = [X_CARD for i in range(ACTOR_COUNT)]
        self.dist = np.zeros(tuple(intents + [X_CARD, Y_CARD]))
        proper_shape = self.dist.shape
        if dist_copy is not None:
            if not (proper_shape == dist_copy.dist.shape):
                raise ValueError("Improper HI_RCT_Dist dimensions on dist_copy with given ACTOR_COUNT and X_DOM")
            self.dist = np.array(dist_copy)
    
    def tell (self, intents, action, reward):
        index = intents + [action, reward]
        self.dist[tuple(index)] += 1
    
    def prob_query (self, i, x, y):
        '''
        Computes P(Y_{x} = y | i)
           - i is given as a dictionary of actor ids mapped to their given
             intent, e.g., {0: 1, 2: 0} is evidence for actors 0 and 2
             experiencing intents 1 and 0, respectively
           - x, an int indicating the counterfactual antecedent treatment
           - y, an int indicating the query outcome
        '''
        matches = self.dist[tuple(i + [x, y])]
        total   = 0
        for j in self.Y_DOM:
            total += self.dist[tuple(i + [x, j])]
        return matches / total
            