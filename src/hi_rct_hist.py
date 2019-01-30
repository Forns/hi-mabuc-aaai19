'''
  hi_rct_hist.py
  
  === Description ===
  Actor and multi-actor history recording class to track reward
  distributions over P(Y_x | I) where I can be the intents of any
  number of actors
  
  [!] Note: inefficient storage solution for HI-rewards,
  which is designed as such only for experimental purposes
  to be easily inspected
'''

import math
import numpy as np
from hi_rct_utl import *
from hi_rct_dist import HI_RCT_Dist

class HI_RCT_Hist:
    
    @staticmethod
    def find_iecs (intents, history, IEC_TOL):
        int_corr = np.corrcoef(intents)
        # iecs will collect the pairs of variables that meet the correlational
        # similarity threshold; e.g., if actors 0 and 1 meet the criteria, their
        # tuple (0, 1) will exist in the iecs list
        iec_pairs = [(i, j) for i in range(1, history.ACTOR_COUNT) for j in range(0, i) if (abs(1 - int_corr[i, j]) < IEC_TOL)]
        # self.iecs is a list of sets grouping indexed actors
        return get_iec_clusters(history.ACTOR_COUNT, iec_pairs)
    
    @staticmethod
    def find_iec_hist(iecs, history):
        '''
        Finds the "average" intent-specific rewards of actors in each IEC
        '''
        N_IECs = len(iecs)
        iec_history = HI_RCT_Hist(N_IECs, history.X_DOM, history.Y_DOM)
        iec_combos = iec_history.actor_combs
        # Each iec is a set composed of actor ids of those belonging to an IEC;
        # IECs themselves are indexed in the list self.iecs
        for iec_dist_ind, iec_combo in enumerate(iec_combos):
            prod_list = [iecs[iec] for iec in iec_combo]
            dist_combos = list(itertools.product(*prod_list))
            dist_list = [history.get_actor_comb_dist(ac) for ac in dist_combos]
            iec_history.history[iec_dist_ind] = HI_RCT_Dist.average_dists(dist_list)
        return iec_history
    
    def __init__ (self, ACTOR_COUNT, X_DOM, Y_DOM):
        self.ACTOR_COUNT = ACTOR_COUNT
        self.X_DOM = X_DOM
        self.Y_DOM = Y_DOM
        self.actor_combs = actor_combos(ACTOR_COUNT)
        self.history = []
        for a in self.actor_combs:
            self.history.append(HI_RCT_Dist(len(a), X_DOM, Y_DOM))
    
    def __get_comb_intents (self, actor_comb, intents):
        results = []
        for i in actor_comb:
            results.append(intents[i])
        return results
    
    def tell (self, intents, action, reward):
        '''
        Updates the history for the given intents, arm, and reward;
        NB, it is expected that |i| = self.ACTOR_COUNT
        '''
        for i, a in enumerate(self.actor_combs):
            self.history[i].tell(self.__get_comb_intents(a, intents), action, reward)
    
    def get_actor_comb_dist (self, actor_comb):
        actor_comb = list(actor_comb)
        actor_comb.sort()
        actor_ind = self.actor_combs.index(tuple(actor_comb))
        return self.history[actor_ind]
    
    def prob_query (self, i, x, y):
        '''
        Computes P(Y_{x} = y | i)
           - i is given as a dictionary of actor ids mapped to their given
             intent, e.g., {0: 1, 2: 0} is evidence for actors 0 and 2
             experiencing intents 1 and 0, respectively
           - x, an int indicating the counterfactual antecedent treatment
           - y, an int indicating the query outcome
        '''
        actor_comb = list(i.keys())
        actor_comb.sort()
        comb_index = self.actor_combs.index(tuple(actor_comb))
        actor_dist = self.history[comb_index]
        int_index  = [i[a] for a in actor_comb]
        return actor_dist.prob_query(int_index, x, y)

