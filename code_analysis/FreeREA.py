import torch
from prune import prune, prune_flops_params
from utils import return_top_k, update_population_history,update_elite
from genotypes import print_genotype
from Exemplar import crossover, sample_exemplar
import random

def search(
        n_iter,
        N=25,
        batch_size = [64],
        input_shape = [3,128,128],
        n=5,
        P=[1],
        R=[1],
        max_flops=200,
        max_params=2.5,
        metrics = ["NASWOT","LogSynFlow"],
        print_search = False
    ):
    """
    :param N:               Size of initial random population
    :param P:               Parallel mutations
    :param R:               Consecutive mutations
    :param max_flops:       Maximum flops (1e6)
    :param max_params:      Maximum parameters 
    :metrics:               You can choiche among "NASWOT","LogSynFlow" and "Skip"
    :print_search:          Whether print search algorithm step by step
    :return:                Current population, history and top 5 architectures.
    """

    # start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_batch = generate_random_batch(batch_size,input_shape,device)
    print(f"Device used: ", random_batch.device )

    # Initialize population of N feasible exemplars
    print("Generating population...")
    population = init_population(N, [1] + input_shape, max_flops, max_params)
    print("Population successfull generated.")

    # Initialize history
    print("Initializing history...")
    history = { exemplar.get_genotype_str() : exemplar for exemplar in population }
    print("History setted up.")

    # Prune according to feasibility, ageing and training free metrics
    print("First Pruning...")
    population, history = prune(population, history, max_flops=max_flops,
                               max_params=max_params, K=N, metrics=metrics, random_batch = random_batch)
    print("First Pruning ended.")
    
    # Initialize elite
    elite = []
    random_batch = generate_random_batch(batch_size,input_shape,device)
    
    step = 0
    print("Search started...")
    for i in range(n_iter):
        if print_search : 
            print("**************************************************")
            print(f"\t Iteration number {i + 1} ")

        suitable = False
        j = 0 
        # Keep generating exemplars until at least one new feasible exemplars is generated
        while not suitable:
            j += 1 
            if print_search : 
                print(f" Attempnt to generate : {j}")
            ## Improved tournament selection

            # Sample n (5) elements from our population
            sampled = random.sample(population, n)

            # Return the two best exemplar among the n selected
            top_1,top_2 = return_top_k(sampled, 2, metrics, random_batch)
            if print_search:
                print(f"1st Best architecture elected")
                print_genotype ( top_1.get_genotype() )
                print(f"Score = {top_1.rank}")
                print(f"2nd Best architecture elected")
                print_genotype( top_2.get_genotype() )
                print(f"Score = {top_2.rank} ")

            # elite pruning
            elite = update_elite(elite,top_1,5,metrics,random_batch)

            # Mutate both of them
            population += [ top_1.mutate(generation = step + 1) ]
            population += [ top_2.mutate(generation = step + 1) ]

            # Crossover
            population += [ crossover(top_1,top_2, generation = step + 1 ) ]

            # Update history
            population, history = update_population_history(population,history,replace=False)
            
            # Feasibility prune. Note: only the three new child can be pruned.
            population, history = prune_flops_params(population,history, max_flops, max_params)

            # if the population is again N, no feasible exemplar have been generated
            suitable = len(population) > N
        
        # Up to this moment, history may contains some not born exemplar
        # since an exemplar is considered bord only if feasible

        # ageing and training free metrics pruning
        population, history = prune(population, history, max_flops=max_flops, max_params=max_params,
                                   K=N, metrics=metrics, feasibility=False, random_batch= random_batch)
        
        step += 1
    print("Search ended...")
    return population,history,elite

def init_population(N, input_shape, max_flops=float('inf'), max_params=float('inf') ):
    population = []
    while len(population) < N:
        candidate = sample_exemplar(input_shape)
        if candidate.is_feasible(max_flops, max_params):
            population.append(candidate)
    return population

def generate_random_batch(batch_size, input_shape,device):
    random_batch = torch.rand(batch_size + input_shape)
    random_batch = random_batch.to(device)
    return random_batch
