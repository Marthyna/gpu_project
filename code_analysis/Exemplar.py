import genotypes
from Network import Network
from metrics import *

class Exemplar:
    def __init__(self,genotype : list , input_shape : list = [1,3,128,128], generation = 0 ) -> None:
        self.genotype = genotype
        self.genotype_str = str ( genotype[0] ) 
        for i in range(1,len(genotype)) : 
            self.genotype_str += "_" + str(genotype[i])
        self.input_shape = input_shape
        self.generation = generation
        self.MACs, self.Nparams = get_macs_and_params(Network(self.genotype),self.input_shape)
        # Metrics
        self.NASWOT = None
        self.LogSynFlow = None
        self.Skip = None

        # rank and born
        self.rank = None
        self.born = False

    def get_genotype(self):
        return self.genotype
    
    def get_genotype_str(self):
        return self.genotype_str
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_MACs(self):
        return self.MACs
    
    def get_Nparams(self):
        return self.Nparams
    
    def is_feasible(self, max_flops=float('inf'), max_params=float('inf')):
        return self.MACs <= max_flops and self.Nparams <= max_params
        
    
    def get_cost_info(self):
        r""" Return a dictionary with MACs (i.e FLOPs) and the number of parameters
        To access them, use the keywords MACs or (FLOPs) or n_params"""
        return { "MACs" : self.get_MACs(), "n_params" : self.get_Nparams() }
    
    def get_network(self):
        return Network(self.genotype)
    
    def mutate(self,generation):
        mutated_genotype = genotypes.mutate(self.genotype)
        return Exemplar( mutated_genotype, self.input_shape, generation= generation )
    
    def belong_to_generation(self,generation):
        return self.generation == generation
    
    def get_generation(self):
        return self.generation

    def set_generation(self,generation):
        self.generation = generation
        return self.generation
    
    def get_born(self):
        return self.born
    
    def set_born(self, born):
        self.set_born = born
        return self.set_born

    def get_rank(self):
        return self.rank
    
    def set_rank(self,rank):
        self.rank = rank
        return self.rank

    def get_metrics(self,batch_input):
        self.compute_metrics(batch_input)
        return {
            "MACs": self.get_MACs,
            "Nparams": self.get_Nparams,
            "NASWOT": self.NASWOT,
            "LogSynFlow": self.LogSynFlow,
            "SKIP": self.Skip
        }

    def get_feasibility_metrics(self):
        return self.get_feasibility_metrics()
    
    def get_eval_metrics(self,batch_input):
        self.compute_metrics(batch_input)
        
        return {
            "NASWOT": self.NASWOT,
            "LogSynFlow": self.LogSynFlow,
            "SKIP" : self.Skip
        }

    def compute_metrics(self,batch_input):
        # NASWOT
        if (self.NASWOT is None) : 
            self.NASWOT = compute_naswot_score(self.get_network(),batch_input)

        #LOGSYNFLOW
        if (self.LogSynFlow is None) : 
            self.LogSynFlow = compute_synflow_per_weight(self.get_network(),batch_input)

        if (self.Skip is None):
            self.Skip = skip(self.genotype) 

def crossover(e1 : Exemplar, e2 : Exemplar, generation):
    child_genotype = genotypes.crossover(e1.get_genotype(),e2.get_genotype())
    return Exemplar(child_genotype,e1.get_input_shape(), generation)

def sample_exemplar(input_shape : list):
    genotype = genotypes.sample_genotype()
    return Exemplar(genotype,input_shape)
