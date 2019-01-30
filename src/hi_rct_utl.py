'''
  hi_rct_util.py
  
  === Description ===
  Utility functions for HI-RCT simulation
'''

import numpy as np
import itertools
from collections import defaultdict

def get_dist_index (bitarray):
    '''
    Converts bitarray to integer digit; used for converting
    UC states to an index in true reward parameter matrix
    '''
    ind = 0
    for bit in bitarray:
        ind = (ind << 1) | bit
    return ind

def actor_combos (num_actors):
    '''
    Returns, for the given number of actors, all possible
    actor combinations (sorted numerically) for the purpose
    of testing different intents
    '''
    results = []
    actor_list = list(range(num_actors))
    for i in range(1,num_actors+1):
        results.extend(itertools.combinations(actor_list, i))
    for c in range(len(results)):
        results[c] = tuple(sorted(results[c]))
    return results

def get_plurality (act_list):
    return max(set(act_list), key=act_list.count)

def dfs(adj_list, visited, vertex, result, key):
    '''
    Collects IEC Tuples using simple dfs lifted from StackOverflow:
    https://stackoverflow.com/questions/42036188/merging-tuples-if-they-have-one-common-element
    Credit to niemmi
    '''
    visited.add(vertex)
    result[key].append(vertex)
    for neighbor in adj_list[vertex]:
        if neighbor not in visited:
            dfs(adj_list, visited, neighbor, result, key)

def get_iec_clusters (ACTOR_COUNT, iec_pairs):
    '''
    Clusters actors into IECs
    '''
    adj_list = defaultdict(list)
    for x, y in iec_pairs:
        adj_list[x].append(y)
        adj_list[y].append(x)
    
    result = defaultdict(list)
    visited = set()
    for vertex in adj_list:
        if vertex not in visited:
            dfs(adj_list, visited, vertex, result, vertex)
    
    result = [set(r) for r in result.values()]
    # Account for IECs with single actors
    for i in range(ACTOR_COUNT):
        found = False
        for iec in result:
            if (i in iec):
                found = True
        if (not found):
            result.append(set([i]))
    
    return result

