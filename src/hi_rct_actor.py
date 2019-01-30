'''
  hi_rct_actor.py
  
  === Description ===
  Actor class that provides an intended decision based on current
  trial's UCs
  
  === Parameters ===
  - P_I: a distribution over intent states given unobserved
    confounders
  - X_DOM, Y_DOM: domains over choice and reward
  
'''

import numpy as np
from hi_rct_utl import *

class HI_RCT_Actor:
    
    def __init__ (self, X_DOM, Y_DOM, P_I):
        self.X_DOM = X_DOM
        self.Y_DOM = Y_DOM
        self.P_I = P_I
    
    def get_intent (self, u):
        '''
        Provides an intent sampled from the actor's P(I | U)
        '''
        uInd = get_dist_index(u)
        return np.random.choice(self.X_DOM, p=self.P_I[:, uInd])
    