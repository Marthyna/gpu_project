from utils import update_population_history, clean_history, return_top_k,update_elite
import numpy as np 

def prune(population : list,
          history : dict ,
          max_flops : float,
          max_params : float ,
          metrics : list,
          random_batch,
          feasibility = True,
          K : int = 0,
        ):

    # Whether real world constraints have to be taken into account.
    if feasibility:
        population, history = prune_flops_params(population, history, max_flops, max_params)
    
    # REA
    population = kill_oldest(population, K)
    population, history = prune_tfm(population, history, K, metrics, random_batch)
    return population, history


def prune_flops_params(population, history, max_flops, max_params):
    population, history = update_population_history(population,history)
    history = clean_history(history, max_params, max_flops)
    return [ exemplar for exemplar in population if exemplar.is_feasible(max_flops,max_params) ], history

def kill_oldest(population, K, R = 1):

    # Get exemplars'generations
    generations = [exemplar.get_generation() for exemplar in population]

    # first iteration (all exemplars belong to generation 0), do not perform ageing 
    if ( not np.any( generations ) ):
        return population
    
    # kill the oldest
    for _ in range(R):
        if len(population) > K + R - 1: # K = N = 25
            oldest = np.argmin(generations)
            population.pop(oldest)
            generations = [exemplar.generation for exemplar in population]

    return population

def prune_tfm(population=None, history=None, K=5, metrics = [], random_batch = []):
    # training free metrics

    # Now, exemplars are going to be evaluated. Thus, if it is the first iteration, they born.
    for exemplar in population:
        exemplar.born = True

    population, history = update_population_history(population, history)

    if K >= len(population):
        # No further pruning needed.
        return population, history

    return return_top_k(population, K, metrics, random_batch), history


    




