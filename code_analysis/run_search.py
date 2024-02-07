import argparse

from FreeREA import search
from genotypes import print_genotype
from Exemplar import Exemplar

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Architecture Search with training free metrics for tiny devices')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of search iterations')
    parser.add_argument('--N', type=int, default=25, help='Population size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[3, 128, 128], help='Input shape')
    parser.add_argument('--n', type=int, default=5, help='Number of sampled exemplars')
    parser.add_argument('--P', type=int, nargs='+', default=[1], help='Parallel mutations')
    parser.add_argument('--R', type=int, nargs='+', default=[1], help='Consecutive mutations')
    parser.add_argument('--max_flops', type=float, default=200, help='Maximum M (million) FLOPs ')
    parser.add_argument('--max_params', type=float, default=2.5, help='Maximum M (million) parameters')
    parser.add_argument('--metrics', type=str, nargs='+', default=['NASWOT', 'LogSynFlow'], help='Metrics')
    parser.add_argument('--print_search', action='store_false', help='Print search steps')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    n_iter = args.n_iter
    N = args.N
    batch_size = [args.batch_size]
    input_shape = args.input_shape
    n = args.n
    P = args.P
    R = args.R
    max_flops = args.max_flops
    max_params = args.max_params
    metrics = args.metrics
    print_search = args.print_search

    _,_,top_architectures = search(n_iter, N, batch_size, input_shape, n, P, R, max_flops, max_params, metrics, print_search)

    top_architectures.reverse()

    print(f"Top 5 architectures founded up to iteration {n_iter}")
    for ith,exemplar in enumerate(top_architectures) :

        print("-------------")
        print(f"Rank #{ith + 1}\n")
        print_genotype(exemplar.get_genotype())
        print("-------------")
    


if __name__ == '__main__':
    main()
