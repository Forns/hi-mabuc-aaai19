'''
  hi_rct_sim.py
  
  === Description ===
  Main simulation workhorse for the Heterogeneous-Intent
  Randomized Clinical Trial Recommender System
  
  === Parameters ===
  - Unobserved Confounder Distributions: defines the priors on
    UCs, U: P(U)
  - True Reward Distribution: defines the causal distribution
    over P(Y | do(X), U)
  - Intent Distributions: define the intent distributions over
    latent causes P(I | U)
  - Sample Size: determines the same size of the training set,
    which will be simulated from parameters above

  === Results ===
  - Excel and Graphical representations of u-regret experienced
    by agents in the parameterized MABUC
  
'''

import numpy as np
import plotly as py
import plotly.graph_objs as go
import time
import multiprocessing
from plotly import tools
from joblib import Parallel, delayed
from hi_rct_utl import *
from hi_rct_lrn import HI_RCT_Learner
from hi_rct_actor import HI_RCT_Actor
from hi_rct_agent import *

# ----------------------------------------------------------------
# Configure Simulation Parameters
# ----------------------------------------------------------------

# UC Params
P_S = np.array(
# S =  0    1
    [0.5, 0.5]
)
P_R = np.array(
# R =  0    1
    [0.5, 0.5]
)
P_U = [P_S, P_R]

# True Reward Params
P_TR = np.array([
# S =  0    0    1    1
# R =  0    1    0    1
    [0.7, 0.8, 0.6, 0.7], # X = 0
    [0.9, 0.7, 0.7, 0.5]  # X = 1
])

# Actor Intent Params
P_I_A0 = np.array([
# I^{A0} = XOR(S,R)
# S =  0    0    1    1
# R =  0    1    0    1
    [1.0, 0.0, 0.0, 1.0], # X = 0
    [0.0, 1.0, 1.0, 0.0]  # X = 1
])
P_I_A1 = np.array([
# I^{A1} = s
# S =  0    0    1    1
# R =  0    1    0    1
    [1.0, 1.0, 0.0, 0.0], # X = 0
    [0.0, 0.0, 1.0, 1.0]  # X = 1
])
P_I_A2 = np.array([
# I^{A2} ~ s
# S =  0    0    1    1
# R =  0    1    0    1
    [0.96, 0.96, 0.04, 0.04], # X = 0
    [0.04, 0.04, 0.96, 0.96]  # X = 1
])
P_I_A3 = np.array([
# I^{A3} ~ XOR(S,R)
# S =  0    0    1    1
# R =  0    1    0    1
    [0.96, 0.04, 0.04, 0.96], # X = 0
    [0.04, 0.96, 0.96, 0.04]  # X = 1
])
P_I = [P_I_A0, P_I_A1, P_I_A2]
P_I_RDT = [P_I_A1, P_I_A2, P_I_A2, P_I_A2, P_I_A2, P_I_A0, P_I_A3, P_I_A3, P_I_A3, P_I_A3]
best_actor_inds = (0, 5) # Used to compare performance against "oracle" agent

U_DOM  = X_DOM = Y_DOM = [0, 1]
U_COUNT = len(P_U)
I_COUNT = len(P_I)

# Sampling and MC Parameters
SIM_NAME = "_10act_iec_samp_w_cal"
N        = 10000   # Training set sample size
IEC_TOL  = 0.10    # IEC Difference Tolerance
CAL_SIZE = 20      # Calibration set size for relevant agents
RDC_N    = 1000    # Number of MC simulations for RDC sim
RDC_T    = 10000   # Number of trials per MC simulation for RDC sim
VERBOSE  = False   # Enables [True] / Disables reporting some features
REP_INT  = 10      # Interval of sims before reporting
N_CORES  = multiprocessing.cpu_count()-1
np.random.seed(0)  # For reproducible results

# ----------------------------------------------------------------
# Simulation Functions
# ----------------------------------------------------------------

def gen_sample ():
    '''
    Simulates the HI-RCT with each unit t consisting of:
    [U_t, {I^{A_i}_t}+, X_t, Y_t]
    '''
    # Draw UC States first
    UCs = np.empty((N, U_COUNT), int)
    for i, u in enumerate(P_U):
        UCs[:, i] = np.random.choice(U_DOM, p=u, size=N)
    
    # Draw Intents, Randomized Treatments, and then Outcomes
    ITOs = np.empty((N, I_COUNT + 2), int)
    for j in range(N):
        u = UCs[j, :]
        u_ind = get_dist_index(u)
        # Intents drawn from state of U
        for i, a in enumerate(P_I):
            ITOs[j, i] = np.random.choice(X_DOM, p=a[:, u_ind])
        # Treatment assigned at random
        x = ITOs[j, I_COUNT] = np.random.choice(X_DOM)
        # Reward Y sampled based on U, X
        ITOs[j, I_COUNT + 1] = np.random.choice(Y_DOM, p=[1 - P_TR[x, u_ind], P_TR[x, u_ind]])
    
    return np.hstack((UCs, ITOs))

def run_sim (actors, agents, n):
    '''
    Runs a single MC iteration of the simulation
    '''
    if (n % REP_INT == 0):
        print("  ...starting %d / %d simulations" % (n, RDC_N))
    AG_COUNT = len(agents)
    ag_reg = np.zeros((AG_COUNT, RDC_T))
    ag_opt = np.zeros((AG_COUNT, RDC_T))
    for a in agents:
        a.clear_hist()
        a.calibrate() # For agents that employ HI-RCT only
        
    # RDC Test-set data created a priori for fair comparison
    UCs = np.empty((RDC_T, len(P_U)), int)
    for i, u in enumerate(P_U):
        UCs[:, i] = np.random.choice(U_DOM, p=u, size=RDC_T)
    for t in range(RDC_T):
        # Get current round's actor intents
        u_t = UCs[t, :]
        u_ind = get_dist_index(u_t)
        # Find the optimal action and reward rate for this t
        best_x_t = np.argmax(P_TR[:, u_ind])
        max_t = P_TR[best_x_t, u_ind]
        i_t = [a.get_intent(u_t) for a in actors]
        # Determine chosen action and reward for each agent
        # within this trial, t
        for a_ind, ag in enumerate(agents):
            x_t = ag.choose(i_t)
            y_t = np.random.choice(Y_DOM, p=[1 - P_TR[x_t, u_ind], P_TR[x_t, u_ind]])
            ag.give_feedback(i_t, x_t, y_t)
            r_t = max_t - y_t
            ag_reg[a_ind, t] += r_t
            ag_opt[a_ind, t] += int(x_t == best_x_t)
                
    return [ag_reg, ag_opt]

def gen_graph (cum_reg, cum_opt, names, colors):
    '''
    Reporting mechanism that generates graphical reports on the
    probability that each agent takes the optimal action and the
    agent's cumulative u-regret, both as a function of the current
    trial
    '''
    AG_COUNT = cum_reg.shape[0]
    traces = []
    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Probability of Optimal Action', 'Cumulative u-Regret'))
    fig['layout']['xaxis1'].update(title='Trial', range=[0, RDC_T])
    fig['layout']['xaxis2'].update(title='Trial', range=[0, RDC_T])
    fig['layout']['yaxis1'].update(title='Probability of Optimal Action')
    fig['layout']['yaxis2'].update(title='Cumulative u-Regret')
    
    # Plot cumulative u-regret
    for a in range(AG_COUNT):
        trace = go.Scatter(
            x = list(range(RDC_T)),
            y = cum_opt[a, :],
            line = dict(
                color = colors[a]
            ),
            name = names[a]
        )
        fig.append_trace(trace, 1, 1)
        
    # Plot optimal arm choice
    for a in range(AG_COUNT):
        trace = go.Scatter(
            x = list(range(RDC_T)),
            y = cum_reg[a, :],
            line = dict(
                color = colors[a]
            ),
            name = "[REG]" + names[a],
            showlegend = False
        )
        fig.append_trace(trace, 1, 2)
    
    py.offline.plot(fig, filename=("./plots/cum_reg" + SIM_NAME + ".html"))


# ----------------------------------------------------------------
# Simulation Workhorse
# ----------------------------------------------------------------

if __name__ == "__main__":
    start_t = time.clock()
    print("=== HI-RCT Simulation Beginning ===")
    
    # NOTE: Can be placed inside of the sim loop to generate each
    # MC iteration:
    # ----------------
    # Generate training data for this run
    complete_data = gen_sample()
    training_data = complete_data[:,2:]
    # Train model on training set to learn IECs
    training_model = HI_RCT_Learner(complete_data, training_data, IEC_TOL, U_DOM, X_DOM, Y_DOM, VERBOSE=VERBOSE)
    # ----------------
    
    # Configure current run's actors
    actors = [HI_RCT_Actor(X_DOM, Y_DOM, a) for a in P_I_RDT]
    
    # Initialize learning agents
    agents = [
        Agent_HI_RDT(training_model, actors),
        Agent_HI_RDT_IEC_Learned(training_model, actors, IEC_TOL),
        Agent_HI_RCT_RDT_Rand(training_model, actors, IEC_TOL, CAL_SIZE),
        Agent_HI_RCT_RDT_Heur(training_model, actors, IEC_TOL, CAL_SIZE),
        Agent_HI_RDT_IEC_Given(training_model, actors, best_actor_inds),
        Agent_HI_RDT_IEC_Given_Cal(training_model, actors, best_actor_inds, IEC_TOL, CAL_SIZE),
    ]
    ag_names = [
        "HI-RDC-A",
        "HI-RDC-L",
        "HI-RDC-RCT-R",
        "HI-RDC-RCT-H",
        "Oracle",
        "Oracle w/ Cal",
    ]
    ag_colors = [
        ('rgb(255, 0, 0)'),
        ('rgb(0, 0, 255)'),
        ('rgb(255, 165, 0)'),
        ('rgb(255, 0, 255)'),
        ('rgb(0, 128, 0)'),
        ('rgb(112, 128, 144)'),
        ('rgb(255, 215, 0)'),
        ('rgb(128, 128, 0)')
    ]
    AG_COUNT = len(agents)
    
    # Record-keeping data structures across simulations
    round_reg = np.zeros((AG_COUNT, RDC_T))
    round_opt = np.zeros((AG_COUNT, RDC_T))
    cum_reg = np.zeros((AG_COUNT, RDC_T))
    cum_reg_rep = np.zeros((AG_COUNT, RDC_N))
    
    # MAIN WORKHORSE:
    sim_results = Parallel(n_jobs=N_CORES, verbose=1)(delayed(run_sim)(actors, agents, i) for i in range(RDC_N))
    for (ind, r) in enumerate(sim_results):
        cum_reg_rep[:, ind] = [np.sum(r[0][i, :]) for i in range(AG_COUNT)]
        round_reg += r[0]
        round_opt += r[1]

    # Reporting phase:
    for a in range(AG_COUNT):
        cum_reg[a] = np.array([np.sum(round_reg[a, 0:i+1]) for i in range(RDC_T)])
    cum_reg = cum_reg / RDC_N
    cum_opt = round_opt / RDC_N
    gen_graph(cum_reg, cum_opt, ag_names, ag_colors)
    np.savetxt("./plots/cum_reg_rep" + SIM_NAME + ".csv", cum_reg_rep, delimiter=",")
    np.savetxt("./plots/cum_reg" + SIM_NAME + ".csv", cum_reg, delimiter=",")
    np.savetxt("./plots/cum_opt" + SIM_NAME + ".csv", cum_opt, delimiter=",")
    
    print("[!] HI-RDC Simulation Completed (%d s)" % (time.clock() - start_t))
    