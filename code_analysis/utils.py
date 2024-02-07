import numpy as np
from genotypes import equal

def update_population_history(population : list, history : dict, replace = True):
    if replace:
        # Update already seen genotypes
        history.update({exemplar.get_genotype_str(): exemplar for exemplar in population})
    else:
        # Add new genotypes
        history.update({exemplar.get_genotype_str(): exemplar for exemplar in population if exemplar.get_genotype_str() not in history})
        # For already seen exemplars, just update the generation
        for exemplar in population:
            history[exemplar.get_genotype_str()].set_generation(exemplar.get_generation())

    # This is also removing exemplars with same genotype from population
    current_genotypes = set([exemplar.get_genotype_str() for exemplar in population])
    population = [history[genotype] for genotype in current_genotypes]
    return population, history

def clean_history(history : dict, max_flops : float, max_params : float):
    return {genotype: exemplar for genotype, exemplar in history.items()
            if exemplar.is_feasible( max_flops = max_flops, max_params = max_params) }

def return_top_k(exemplars, K, metric_names, batch_input) :
    exemplars = [exemplar for exemplar in exemplars if exemplar.born]
     
    values_dict = {}

    # compute metrics
    for metric in metric_names : 
        values_dict[metric] = np.array( [exemplar.get_eval_metrics(batch_input)[metric] for exemplar in exemplars ])
    
    # initialize to zero
    scores = np.zeros(len(exemplars))

    scores_dict = {}

    # get score
    for metric_n in metric_names:
        scores_dict[metric_n] = values_dict[metric_n] / (np.max(np.abs(values_dict[metric_n])) + 1e-9)
        scores += scores_dict[metric_n]

    # set rank 
    for _, (exemplar, rank) in enumerate(zip(exemplars, scores)):
        exemplar.rank = rank
    
    # The higher the better
    exemplars.sort(key=lambda x: -x.rank)

    return exemplars[:K]

def update_elite(elite : list, candidate,K, metrics,batch_input):
    propose = elite + [candidate]
    top_1 = return_top_k(propose, K = 1, metric_names=metrics,batch_input=batch_input)[0]
    top_1_genotype = top_1.get_genotype()
    candidate_genotype = candidate.get_genotype()
    elite_genotypes = [ e.get_genotype() for e in elite ]
    if ( equal(top_1_genotype,candidate_genotype) and top_1_genotype not in elite_genotypes ) :
        if (len(propose) > K ):
            propose.pop(0) 
        return propose
    return elite