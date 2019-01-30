'''
  hi_rct_sim.py
  
  === Description ===
  Learning system for HI-RCT training sets and HI test sets 
  
  === Inputs ===
  - Training Set: data consisting of:
      - Experimental Data: records of a randomized clinical trial
        in which each datum corresponds to a particular unit's
        randomly assigned treatment and its outcome
      - Training Actor Intents: for each record in the Exp. Data,
        intents of actors are collected with the objective of
        determining the number of intent equivalence classes (IECs)
  - Training set format: [{Intents}, Assigned Treatment, Outcome]
  
  === Output ===
  Recommender Model trained on HI-RCT data that, for any new
  test unit, will maximize HI-intent-specific treatment efficacy
  given any subset of IEC intents
'''

import numpy as np
import itertools
import queue
import random
from collections import Counter
from scipy.special import comb
from hi_rct_utl import *
from hi_rct_hist import HI_RCT_Hist
from hi_rct_dist import HI_RCT_Dist

class HI_RCT_Learner:
    
    def __init__ (self, complete_data, training_set, IEC_TOL, U_DOM, X_DOM, Y_DOM, VERBOSE=True):
        # Note: the complete data is never accessed explicitly, except to support some
        # agents under which we assume that a given unit's / trial's UC state needs to be
        # fed to an actor's intent function during calibration
        self.complete_data = complete_data
        self.training_set = training_set
        self.ACTOR_COUNT = training_set.shape[1] - 2
        self.IEC_TOL = IEC_TOL
        self.X_DOM = X_DOM
        self.Y_DOM = Y_DOM
        self.U_DOM = U_DOM
        self.VERBOSE = VERBOSE
        ACTOR_COUNT = self.ACTOR_COUNT
        TRAINING_N = training_set.shape[0]
        X_CARD = len(X_DOM)
        Y_CARD = len(Y_DOM)
        
        # Parameter learning
        self.history = HI_RCT_Hist(ACTOR_COUNT, X_DOM, Y_DOM)
        for d in training_set:
            i_t = d[:ACTOR_COUNT]
            x_t = d[ACTOR_COUNT]
            y_t = d[-1]
            self.history.tell(i_t, x_t, y_t)
        
        # IEC Model learning phase
        self.iecs = HI_RCT_Hist.find_iecs(np.transpose(self.training_set[:, 0:self.ACTOR_COUNT]), self.history, IEC_TOL)
        self.iec_history = HI_RCT_Hist(len(self.iecs), X_DOM, Y_DOM)
        for d in training_set:
            i_t = d[:ACTOR_COUNT]
            x_t = d[ACTOR_COUNT]
            y_t = d[-1]
            actor_choices = [[i_t[a] for a in iec] for iec in self.iecs]
            iec_choices = [Counter(a).most_common(1)[0][0] for a in actor_choices]
            self.iec_history.tell(iec_choices, x_t, y_t)
            
        print(self.iecs)
        
    def __get_intent_corr (self):
        '''
        Called after loading training set to find the correlation matrix between
        actor intents
        '''
        return np.corrcoef(np.transpose(self.training_set[:, 0:self.ACTOR_COUNT]))
    
    def __find_iecs (self):
        '''
        Finds the intent equivalence classes (IECs) of actors in the training
        set, and then clusters actors by their given index in the self.iecs attr
        '''
        # Generate intent correlation matrix
        int_corr = self.__get_intent_corr()
        # iecs will collect the pairs of variables that meet the correlational
        # similarity threshold; e.g., if actors 0 and 1 meet the criteria, their
        # tuple (0, 1) will exist in the iecs list
        iec_pairs = [(i, j) for i in range(1, self.ACTOR_COUNT) for j in range(0, i) if (abs(1 - int_corr[i, j]) < self.IEC_TOL)]
        # self.iecs is a list of sets grouping indexed actors
        self.iecs = iecs = get_iecs(self.ACTOR_COUNT, iec_pairs)
        self.__find_iec_hist()
        
        if (self.VERBOSE):
            print("  [I] IEC Comparison Complete:")
            print("    [>] Detected %i IEC%s between actors: %s" % (len(iecs), "" if (len(iecs) == 1) else "s", iecs))
    
    def get_calibration_samples (self, CAL_SIZE):
        '''
        Returns calibration samples from the complete dataset based on the calibration
        heuristic; returns the requested number of samples
        '''
        ACTOR_COUNT = self.ACTOR_COUNT
        U_COUNT = len(self.U_DOM) # Note: used only for indexing in the dataset
        iecs = self.iecs
        
        # Heuristic DS setup
        best_samples = queue.PriorityQueue()
        int_uniqueness_hist = dict()
        undo_mem = dict()
        
        # Note: iterating through the complete dataset (UCs included) for simulation
        # purposes only; this heuristic technique does not ever address the UC states
        random.shuffle(self.complete_data)
        t = 0
        for d in self.complete_data:
            undo_mem[t] = dict()
            h_score = 0
            i_t = d[U_COUNT:U_COUNT + ACTOR_COUNT]
            x_t = d[-2]
            y_t = d[-1]
            
            # Heuristic component A: intra-iec agreement
            actor_choices = [[i_t[a] for a in iec] for iec in iecs]
            iec_choices = [Counter(a).most_common(1)[0][0] for a in actor_choices]
            intra_iec_agree = np.zeros(len(iecs))
            for c_ind in range(len(iecs)):
                c = actor_choices[c_ind]
                count = 0
                for i in c:
                    count += i == iec_choices[c_ind]
                intra_iec_agree[c_ind] = count / len(c)
            h_score += np.average(intra_iec_agree)
            
            # Heuristic component B: uniqueness of intent-action-reward
            # in current sample
            # int_tup = tuple(iec_choices)
            int_tup = tuple(iec_choices + [x_t])
            undo_mem[t]["int"] = int_tup
            if (not int_tup in int_uniqueness_hist):
                int_uniqueness_hist[int_tup] = 0
            int_uniqueness_hist[int_tup] += 1
            h_score += 1 / int_uniqueness_hist[int_tup]
            
            # Heuristic component C: biasing towards optimal
            # action under given iec condition
            action_rewards = [self.iec_history.prob_query(dict(enumerate(iec_choices)), x, 1) for x in self.X_DOM]
            h_score += (max(action_rewards) == action_rewards[x_t] and y_t == 1) or (max(action_rewards) != action_rewards[x_t] and y_t == 0)
            
            added_samp = new_samp = (h_score, t, d)
            if best_samples.qsize() >= CAL_SIZE:
                worst_samp = best_samples.get()
                added_samp = max(worst_samp, new_samp)
                worse_samp = min(worst_samp, new_samp)
                # Sample tuple [1] has t
                int_uniqueness_hist[undo_mem[worse_samp[1]]["int"]] -= 1
                
            best_samples.put(added_samp)
            t += 1
            
        
        result = []
        while (not best_samples.empty()):
            result.append(best_samples.get())
        print(result)
        return [r[-1] for r in result]
        
