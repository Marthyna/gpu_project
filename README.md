- Goals:
1. Try to optimize the already designed Neural Network on GPU (all of the matrices computations) for Ensimag CUDA machines. If not the full machine, then maybe a set of its computations

PLAN A:
2. Run the same machine ops on CPU 
3. Run the same machine with "cuda" option in Python
4. Run our own version
5. Compare the results

PLAN B:
2. Figure out how much computational resource (nb FLOPS) does the optimized version of the network use
3. This usage is the lower bound to be respected if a tiny device wants to use our network

