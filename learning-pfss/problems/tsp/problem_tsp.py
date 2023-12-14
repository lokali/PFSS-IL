from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search
from problems.tsp.neh_run import neh 
from problems.tsp.scheduling import schedule_time_Batch_1

np.random.seed(1)

def nearest_neighbor_graph(nodes, neighbors, knn_strat):
    """Returns k-Nearest Neighbor graph as a **NEGATIVE** adjacency matrix
    """
    num_nodes = len(nodes)
    # If `neighbors` is a percentage, convert to int
    if knn_strat == 'percentage':
        neighbors = int(num_nodes * neighbors)
    
    if neighbors >= num_nodes-1 or neighbors == -1:
        W = np.zeros((num_nodes, num_nodes))
    else:
        # Compute distance matrix
        W_val = squareform(pdist(nodes, metric='euclidean'))
        W = np.ones((num_nodes, num_nodes))
        
        # Determine k-nearest neighbors for each node
        knns = np.argpartition(W_val, kth=neighbors, axis=-1)[:, neighbors::-1]
        # Make connections
        for idx in range(num_nodes):
            W[idx][knns[idx]] = 0
    
    # Remove self-connections
    np.fill_diagonal(W, 1)
    return W


def tour_nodes_to_W(tour_nodes):
    """Computes edge adjacency matrix representation of tour
    """
    num_nodes = len(tour_nodes)
    tour_edges = np.zeros((num_nodes, num_nodes))
    for idx in range(len(tour_nodes) - 1):
        i = tour_nodes[idx]
        j = tour_nodes[idx + 1]
        tour_edges[i][j] = 1
        tour_edges[j][i] = 1
    # Add final connection
    tour_edges[j][tour_nodes[0]] = 1
    tour_edges[tour_nodes[0]][j] = 1
    return tour_edges


class TSP(object):
    """Class representing the Travelling Salesman Problem
    """

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        """Returns TSP tour length for given graph nodes and tour permutations

        Args:
            dataset: graph nodes (torch.Tensor)
            pi: node permutations representing tours (torch.Tensor)

        Returns:
            TSP tour length, None
        """
        # Check that tours are valid, i.e. contain 0 to n -1

        # assert (
        #     torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
        #     pi.data.sort(1)[0]
        # ).all(), "Invalid tour:\n{}\n{}".format(dataset, pi)

        # # Gather dataset in order of tour
        # d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        # return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None
        makespan, D = schedule_time_Batch_1(dataset, pi) # shape(makespan)= Batch * 1, shape(D)=Batch * n_machine
        cost = torch.Tensor(makespan)
        # D = torch.Tensor(D)
        return cost, None 

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(nodes, graph, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        """Method to call beam search, given TSP samples and a model
        """

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(nodes, graph)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            nodes, graph, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)
    
    
class TSPSL(TSP):
    """Class representing the Travelling Salesman Problem, trained with Supervised Learning
    """

    NAME = 'tspsl'


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, min_size=20, machine=5, max_size=50, batch_size=128,
                 num_samples=128000, offset=0, distribution=None, neighbors=20, 
                 knn_strat=None, supervised=False, nar=False):
        """Class representing a PyTorch dataset of TSP instances, which is fed to a dataloader

        Args:
            filename: File path to read from (for SL)
            min_size: Minimum TSP size to generate (for RL)
            max_size: Maximum TSP size to generate (for RL)
            batch_size: Batch size for data loading/batching
            num_samples: Total number of samples in dataset
            offset: Offset for loading from file
            distribution: Data distribution for generation (unused)
            neighbors: Number of neighbors for k-NN graph computation
            knn_strat: Strategy for computing k-NN graphs ('percentage'/'standard')
            supervised: Flag to enable supervised learning
            nar: Flag to indicate Non-autoregressive decoding scheme, which uses edge-level groundtruth

        Notes:
            `batch_size` is important to fix across dataset and dataloader,
            as we are dealing with TSP graphs of variable sizes. To enable
            efficient training without DGL/PyG style sparse graph libraries,
            we ensure that each batch contains dense graphs of the same size.
        """
        super(TSPDataset, self).__init__()

        self.filename = filename
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.offset = offset
        self.distribution = distribution
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.supervised = supervised
        self.nar = nar
        self.machine = machine 


        # This is the data distribution. 
        # print(f"samples: {num_samples}; Distribution: {distribution}")
        # if distribution=='rand':
        #     self.nodes_coords = [torch.FloatTensor(min_size, 5).uniform_(0, 1) for i in range(num_samples)]  # random generator 
        # if distribution=='rand100':
        #     self.nodes_coords = [torch.Tensor(np.random.randint(100, size=(min_size, 5))) for i in range(num_samples)]  # random generator 
        # elif distribution=='chi':
        #     print('this is chi!!!')
        #     self.nodes_coords = [torch.Tensor(np.random.chisquare(1, size=(min_size, 10))) for i in range(num_samples)]
        # elif distribution=='beta13':
        #     self.nodes_coords = [torch.Tensor(np.random.beta(1, 3, size=(min_size, 5))) for i in range(num_samples)]
        # elif distribution=='beta51':
        #     self.nodes_coords = [torch.Tensor(np.random.beta(5, 1, size=(min_size, 5))) for i in range(num_samples)]
        # elif distribution=='normal':
        #     print("this is normal distribution.")
        #     self.nodes_coords = [torch.Tensor(np.random.normal(6, 1, size=(min_size, 5))) for i in range(num_samples)]
        #     for i in range(num_samples):
        #         for j in range(min_size):
        #             for k in range(5):
        #                 if self.nodes_coords[i][j,k] <= 0:
        #                     self.nodes_coords[i][j,k] = 0
        # elif distribution=='tai':
        #     print(os.getcwd())
        #     data = taillards()
        #     self.nodes_coords = [torch.Tensor(np.transpose(i)) for i in data]
        
        # elif distribution=='vrf':
        #     dataset = []
        #     for i in range(10):
        #         data = VRF(60,i)
        #         data = torch.Tensor(data) 
        #         dataset.append(data)
        #     self.nodes_coords = dataset 
            
        if filename is not None:
            self.nodes_coords = []
            self.tour_nodes = []
            print('\nGenerating {} samples of PFSS: job-{}, machine-{}.'.format(num_samples, min_size, machine))
            # self.nodes_coords = np.random.random([num_samples, min_size, 5])
            self.nodes_coords = np.random.chisquare(1, size=(num_samples, min_size, machine)) 
            for i in tqdm(range(num_samples)):
                data = self.nodes_coords[i]
                ms, seq = neh(data)
                self.tour_nodes.append(seq)


        # Generating random TSP samples (usually used for Reinforcement Learning)
        else:
            # Sample points randomly in [0, 1] square
            self.nodes_coords = []

            print('\nGenerating {} samples of TSP{}-{}...'.format(num_samples, min_size, max_size))
            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                # Each mini-batch contains graphs of the same size
                # Graph size is sampled randomly between min and max size
                num_nodes = np.random.randint(low=min_size, high=max_size+1)
                self.nodes_coords += list(np.random.random([batch_size, num_nodes, 5]))
        
        self.size = len(self.nodes_coords)
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        nodes = self.nodes_coords[idx]
        item = {
            'nodes': torch.FloatTensor(nodes),
            'graph': torch.ByteTensor(nearest_neighbor_graph(nodes, self.neighbors, self.knn_strat))
        }
        if self.supervised:
            # Add groundtruth labels in case of SL
            tour_nodes = self.tour_nodes[idx]
            item['tour_nodes'] = torch.LongTensor(tour_nodes)
            if self.nar:
                # Groundtruth for NAR decoders is the TSP tour in adjacency matrix format
                item['tour_edges'] = torch.LongTensor(tour_nodes_to_W(tour_nodes))

        return item
