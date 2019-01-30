'''
  hi_rct_agent.py
  
  === Description ===
  Agent recommender system that learns actor IECs and makes
  hi-regret minimizing choice suggestions based on trials in
  a dynamic, HI-MABUC experiment
  
  === Parameters ===
  - training_model: the HI-RCT model trained on the data from
    a HI-RCT
  - actors: a list of HI_RCT_Actors to simulate actors providing
    intents in the HI-MABUC dynamic experiment
  
'''

import random
import numpy as np
from scipy.stats import chisquare
from hi_rct_utl import *
from hi_rct_hist import HI_RCT_Hist
from hi_rct_actor import *
from hi_rct_lrn import *

# ----------------------------------------------------------------
# Agent Superclass
# ----------------------------------------------------------------

class HI_RCT_Agent:
    
    def __init__ (self, training_model, actors):
        self.X_DOM = training_model.X_DOM
        self.Y_DOM = training_model.Y_DOM
        self.actors = actors
        self.ACTOR_COUNT = len(actors)
        self.training_model = training_model
        self.iecs = [None for a in actors]
        self.intents = []
        self.clear_hist()
    
    def clear_hist (self):
        self.history = HI_RCT_Hist(self.ACTOR_COUNT, self.training_model.X_DOM, self.training_model.Y_DOM)
    
    def calibrate (self):
        '''
        Default calibration behavior: none! Individual agents will
        override this depending on their setup
        '''
        return
    
    def add_cal_data (self, training_sample):
        '''
        Used by calibration agents to add calibrating samples to
        their history
        '''
        training_model = self.training_model
        
        # Collect this agent's managed actors' intents in training_sample, then
        # update history accordingly
        U_CARD = len(training_model.U_DOM)
        for d in training_sample:
            u_t = d[:U_CARD]
            i_t = [a.get_intent(u_t) for a in self.actors]
            x_t = d[-2]
            y_t = d[-1]
            self.intents.append(i_t)
            self.give_feedback(i_t, x_t, y_t)
    
    def give_feedback (self, i, x, y):
        self.history.tell(i, x, y)

# ----------------------------------------------------------------
# Agent Subclasses (specific behaviors)
# ----------------------------------------------------------------
    
class Agent_Exp(HI_RCT_Agent):
    '''
    Agent that maximizes the experimental distribution
    '''
    def __init__ (self, training_model, actors):
        HI_RCT_Agent.__init__(self, training_model, actors)
    
    # Experimental maximizer ignores i
    def choose (self, i):
        return np.random.choice(self.X_DOM)

class Agent_RDT(HI_RCT_Agent):
    '''
    Agent that maximizes a single actor's intent-specific rewards, as
    given by the actor_id in the constructor
    '''
    def __init__ (self, training_model, actors, actor_id):
        HI_RCT_Agent.__init__(self, training_model, actors)
        self.actor_id = actor_id
    
    def choose (self, i):
        rel_dist = self.history.history[self.actor_id].dist
        arm_scores = [np.random.dirichlet(rel_dist[i[self.actor_id], x, :] + 1)[1] for x in self.X_DOM]
        return np.argmax(arm_scores)
        
class Agent_HI_RDT(HI_RCT_Agent):
    '''
    Agent that maximizes by the intents of *all* actors in the
    environment, regardless of their IEC
    '''
    def __init__ (self, training_model, actors):
        HI_RCT_Agent.__init__(self, training_model, actors)
    
    def choose (self, i):
        rel_dist = self.history.history[-1].dist
        arm_scores = [np.random.dirichlet(rel_dist[tuple(i + [x])] + 1)[1] for x in self.X_DOM]
        return np.argmax(arm_scores)

class Agent_HI_RDT_IEC_Learned(HI_RCT_Agent):
    '''
    Agent that maximizes by the intents of those known (a priori) to
    be the relevant IECs of actors in the environment
    '''
    def __init__ (self, training_model, actors, IEC_TOL):
        HI_RCT_Agent.__init__(self, training_model, actors)
        self.IEC_TOL = IEC_TOL
        self.rel_hist = dict()
        self.last_choices = None
    
    def give_feedback (self, i, x, y):
        # Update history
        self.history.tell(i, x, y)
        
        # Update reliability history
        rh = self.rel_hist
        if (self.last_choices != None):
            for iec_combo, iec_act in self.last_choices.items():
                iec_intents = [i[a] for a in iec_combo]
                rel_key = iec_combo
                if (not rel_key in rh):
                    rh[rel_key] = 0
                # Add to the reliability of a combo if its choice was the same
                # as the enacted, and a reward was received
                # Note: this technique is experimental and can be tuned in future
                # study
                rh[rel_key] += ((x == iec_act) and y == 1) or ((x != iec_act) and y == 0)
        
    def choose (self, i):
        '''
        Makes an arm choice via HI-RDC by weighting unreliable intent actors
        less than more reliable ones; each intent combo gets a vote on the arm
        to choose according to their past reliability, culminating in a pulled arm
        that is the most popular
        '''
        self.intents.append(i)
        self.iecs = iecs = HI_RCT_Hist.find_iecs(np.transpose(np.array(self.intents)), self.history, self.IEC_TOL)
        iec_pluralities = [get_plurality(iec_i) for iec_i in [[i[a] for a in iec] for iec in self.iecs]]
        possible_dist_combs = list(itertools.product(*iecs))
        iec_samples = []
        confidence = []
        combo_report = dict()
        votes = np.zeros(len(self.X_DOM))
        for d in possible_dist_combs:
            d = sorted(d)
            rel_dist = self.history.get_actor_comb_dist(d).dist
            iec_samp = np.argmax([(np.random.dirichlet(rel_dist[tuple(iec_pluralities + [x])] + 1)[1]) for x in self.X_DOM])
            iec_samples.append(iec_samp)
            combo_report[tuple(d)] = iec_samp
            vote_weight = 1
            rel_key = tuple(d)
            if (rel_key in self.rel_hist):
                vote_weight = self.rel_hist[rel_key]
            votes[iec_samp] += vote_weight
            confidence.append((vote_weight, iec_samp))
        
        self.last_choices = combo_report
        confidence.sort()
        most_reliable = [x[1] for x in confidence[-7:]]
        return get_plurality(most_reliable)
    
class Agent_HI_RDT_IEC_Given(HI_RCT_Agent):
    '''
    Oracle: Agent that maximizes by the intents of those known (a priori) to
    be the relevant IECs of actors in the environment
    '''
    def __init__ (self, training_model, actors, best_actor_inds):
        HI_RCT_Agent.__init__(self, training_model, actors)
        self.rel_tup = best_actor_inds
    
    def choose (self, i):
        rel_tup  = self.rel_tup
        rel_dist = self.history.get_actor_comb_dist(rel_tup).dist
        arm_scores = [np.random.dirichlet(rel_dist[tuple([i[rel_tup[0]]] + [i[rel_tup[1]]] + [x])] + 1)[1] for x in self.X_DOM]
        return np.argmax(arm_scores)

class Agent_HI_RDT_IEC_Given_Cal(Agent_HI_RDT_IEC_Given):
    '''
    Oracle w/ Cal: Agent that maximizes by the intents of those known (a priori) to
    be the relevant IECs of actors in the environment, plus starts with some free
    samples from the calibration set
    '''
    def __init__ (self, training_model, actors, best_actor_inds, IEC_TOL, CAL_SIZE):
        Agent_HI_RDT_IEC_Given.__init__(self, training_model, actors, best_actor_inds)
        self.IEC_TOL = IEC_TOL
        self.CAL_SIZE = CAL_SIZE
    
    def calibrate (self):
        training_model = self.training_model
        actors = self.actors
        
        # Firstly, must score each of the datasets units by the
        # calibration heuristic, retrieving the CAL_SIZE most
        # informative; this is done by the training_model
        calibration_samples = self.training_model.get_calibration_samples(self.CAL_SIZE)
        self.add_cal_data(calibration_samples)
    
class Agent_HI_RCT_RDT_Rand(Agent_HI_RDT_IEC_Learned):
    '''
    Agent that attempts to both learn the IECs of actors in the
    environment, and then maximize the IEC-specific reward
    '''
    def __init__ (self, training_model, actors, IEC_TOL, CAL_SIZE):
        Agent_HI_RDT_IEC_Learned.__init__(self, training_model, actors, IEC_TOL)
        self.CAL_SIZE = CAL_SIZE
    
    def calibrate (self):
        '''
        Calibration Phase: this agent randomly samples from trials / units in the
        HI-RCT dataset to pre-sample the actors' IECs and intent-specific rewards
        without cost in real-time mistakes
        '''
        training_model = self.training_model
        actors = self.actors
        # Gather some CAL_SIZE number of random HI-RCT sample units
        idx = np.random.randint(len(training_model.complete_data), size=self.CAL_SIZE)
        calibration_samples = training_model.complete_data[idx, :]
        self.add_cal_data(calibration_samples)

class Agent_HI_RCT_RDT_Heur(Agent_HI_RDT_IEC_Learned):
    '''
    Agent that attempts to both learn the IECs of actors in the
    environment, and then maximize the IEC-specific reward
    '''
    def __init__ (self, training_model, actors, IEC_TOL, CAL_SIZE):
        Agent_HI_RDT_IEC_Learned.__init__(self, training_model, actors, IEC_TOL)
        self.CAL_SIZE = CAL_SIZE
    
    def calibrate (self):
        '''
        Calibration Phase: this agent samples from trials / units in the
        HI-RCT dataset to pre-sample the actors' IECs and intent-specific rewards
        without cost in real-time mistakes; done according to heuristic
        described in paper
        '''
        training_model = self.training_model
        actors = self.actors
        
        # Firstly, must score each of the datasets units by the
        # calibration heuristic, retrieving the CAL_SIZE most
        # informative; this is done by the training_model
        calibration_samples = self.training_model.get_calibration_samples(self.CAL_SIZE)
        self.add_cal_data(calibration_samples)
        