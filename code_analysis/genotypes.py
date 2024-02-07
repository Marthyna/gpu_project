import numpy as np
from modalities import * 

# We can have 6 type of different cells
cell = ["ConvNext", "InvBottleNeck"]
kernel_size = [3,5,7]

def sample_cell():
    # Sample one cell
    op = cell[ np.random.binomial(n =1 , p=0.5) ]
    k = kernel_size[ np.argmax ( np.random.multinomial(n=1, pvals = [1/len(kernel_size)] * len(kernel_size) ) ) ]
    return [ op , k ]

# There are two different cells and 2^3 - 1 (0,0,0) different types of connections.
# Therefore, there are 6 * 6 * 7 =  288 different blocks.
# However, some blocks are isomorphe. Therefore the number of different blocks is 180
def sample_block():
    # sample cell 1
    op1 = sample_cell()
    # sample cell 2
    op2 = sample_cell()
    # whether include skip connection or not
    skip = np.random.binomial(n = 1, p = 0.5 )
    # whether include previous or not
    prev = np.random.binomial(n = 1, p = 0.5 )
    # whether include input or not
    input = np.random.binomial(n = 1, p = 0.5 )

    if ( not np.any([skip,prev,input]) ):
        skip, prev, input = np.random.multinomial( n = 1, pvals = [1/3, 1/3, 1/3] )
    return [ op1, op2, [skip, prev, input] ]

# The number of different possible genotypes is 186^4 = 1.196.883.216
def sample_genotype():
    # Stage 1 
    block_1 = sample_block()

    # Stage 2
    block_2 = sample_block()

    # Stage 3
    block_3 = sample_block()

    # Stage 4
    block_4 = sample_block()

    # Build genotype
    genotype = [block_1, block_2, block_3, block_4]

    return genotype


def mutate(genotype):
    # Stage containg the block to be mutated. Uniformly sampled
    block_choice = np.random.multinomial(1, np.array([1]*4 ) / 4 )
    # Just get the one with outcome 1
    mutating_block_id = np.argmax(block_choice)

    # Mutate
    old_gene = genotype[mutating_block_id]
    different = False
    while ( not different ):
        # sample a gene
        new_gene = sample_block()
        # if it is equal to the actual gene, discard this mutation.
        different = not equal_block(new_gene,old_gene)

    new_genotype = [ *genotype ]
    new_genotype[mutating_block_id] = new_gene
    return new_genotype

def crossover(genotype1, genotype2):
    # Create copies of the genotypes to avoid modifying the original ones
    new_genotype1 = [*genotype1]
    new_genotype2 = [*genotype2]

    # Choose a random stage to perform crossover
    crossover_stage = np.random.randint(0, 4)

    # Perform crossover for the selected stage
    new_genotype1[crossover_stage] = genotype2[crossover_stage]
    new_genotype2[crossover_stage] = genotype1[crossover_stage]
    candidates = [new_genotype1, new_genotype2 ]

     # Randomly choiche one of the two genotypes
    return candidates [ np.random.binomial(1,0.5) ]


def equal(genotype_1, genotype_2) -> bool:
    """True if all blocks are equals"""
    eq = []
    for block_1, block_2 in zip (genotype_1,genotype_2):
        eq.append(equal_block(block_1,block_2))
    return np.all(eq)

def equal_block(block_1,block_2) -> bool :
    info_1 = get_connection_dictionary ( block_1[2] ) 
    info_2 = get_connection_dictionary ( block_2[2] )
    result = False

    if one_operation_mode(info_1) and one_operation_mode(info_2):
        result = block_1[0] == block_2[0]

    if two_branch_mode(info_1) and two_branch_mode(info_2):
        result = block_1[0] == block_2[0] and block_1[1] == block_2[1]

    if sequential_mode(info_1) and sequential_mode(info_2):
        result = block_1[0] == block_2[0] and block_1[1] == block_2[1] and info_1["skip"] == info_2["skip"]
    
    if complete_mode(info_1) and complete_mode(info_2):
        result = block_1[0] == block_2[0] and block_1[1] == block_2[1] and info_1["skip"] == info_2["skip"]
    
    # Sequential and operational mode isomorphism
    
    if one_operation_mode(info_1) and sequential_mode(info_2) and info_2["skip"] == 0:
        if block_2[0] == block_2[1] :
            result = block_1[0] == block_2[0]
    
    if one_operation_mode(info_2) and sequential_mode(info_1) and info_1["skip"] == 0:
        if block_1[0] == block_1[1] :
            result = block_1[0] == block_2[0]
    
    # Default: if two blocks are connected differently, then they are not equal
    return result
                
def get_connection_dictionary(connection_info : list) -> list:
    return {
        "skip":connection_info[0],
        "prev":connection_info[1],
        "input":connection_info[2],
    }
    
def print_genotypes(genotypes : list ):
    for j, genotype in enumerate( genotypes )  :
        print(f"Genotype {j + 1}")
        print_genotype(genotype)



def print_genotype(genotype : list):
        for i,block in enumerate(genotype):
            str = ""
            print(f"\t\t**** STAGE {i + 1} ****")
            cell_1, cell_2, connection_info = block[0],block[1], get_connection_dictionary( block[2] )
            str += f"{cell_1} --- "
            print_second_cell = True
            print_skip = True
            if( one_operation_mode(connection_info) ):
                # Do nothing 
                str += "One operation mode." 
                print_second_cell = False
                print_skip = False
            if ( two_branch_mode(connection_info)):
                # Two branches case
                str += "Two branches mode --- "
                print_skip = False
            if(sequential_mode(connection_info) ):
                # Sequential case
                str += "Sequential mode --- "
            if ( complete_mode(connection_info)):
                str += " complete mode --- "
            if(print_second_cell):
                str += f"{cell_2}."
            if connection_info["skip"] and print_skip:
                str += " SKIP ON."    
            print(str)

            

        