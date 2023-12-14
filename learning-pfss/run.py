#!/usr/bin/env python

import os
import json
import pprint as pp
import numpy as np

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from options import get_options
from train import train_epoch, train_epoch_sl, validate, get_inner_model

from nets.attention_model import AttentionModel
from nets.nar_model import NARModel
from nets.critic_network import CriticNetwork
from nets.encoders.gat_encoder import GraphAttentionEncoder
from nets.encoders.gnn_encoder import GNNEncoder
from nets.encoders.mlp_encoder import MLPEncoder

from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline

from utils import torch_load_cpu, load_problem
import time 
import warnings
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.")


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(
            opts.log_dir, "{}_{}-{}".format(opts.problem, opts.min_size, opts.max_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)
    assert opts.problem == 'tspsl', "Only TSP is supported for supervised learning"

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if opts.eval_only: 
        load_path = opts.model 
        opts.model = 'attention'
    if load_path is not None:
        print('\nLoading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'nar': NARModel,
        # 'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    encoder_class = {
        'gnn': GNNEncoder,
        'gat': GraphAttentionEncoder,
        'mlp': MLPEncoder
    }.get(opts.encoder, None)
    assert encoder_class is not None, "Unknown encoder: {}".format(encoder_class)
    model = model_class(
        problem=problem,
        machine=opts.machine, 
        embedding_dim=opts.embedding_dim,
        encoder_class=encoder_class,
        n_encode_layers=opts.n_encode_layers,
        aggregation=opts.aggregation,
        aggregation_graph=opts.aggregation_graph,
        normalization=opts.normalization,
        learn_norm=opts.learn_norm,
        track_norm=opts.track_norm,
        gated=opts.gated,
        n_heads=opts.n_heads,
        tanh_clipping=opts.tanh_clipping,
        mask_inner=True,
        mask_logits=True,
        mask_graph=False,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    # Compute number of network parameters
    nb_param = 0
    for param in model.parameters():
        nb_param += np.prod(list(param.data.size()))
    # print('Number of parameters: ', nb_param)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': opts.lr_model}])

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Load/generate datasets
    val_datasets = problem.make_dataset(
                filename='il', min_size=opts.min_size, machine=opts.machine, batch_size=opts.batch_size, num_samples=opts.val_size, 
                neighbors=opts.neighbors, knn_strat=opts.knn_strat, supervised=True, nar=False
            )

    if opts.eval_only:
        start = time.time()
        validate(model, val_datasets, problem, opts)
        end = time.time()
        print(f"eval cost time: {end-start}")
    else:
        train_dataset = problem.make_dataset(
            filename='il', min_size=opts.min_size, machine=opts.machine, batch_size=opts.batch_size, num_samples=opts.epoch_size, 
            neighbors=opts.neighbors, knn_strat=opts.knn_strat, supervised=True, nar=False)
        opts.epoch_size = train_dataset.size  # Training set size might be different from specified epoch size

        # Start training loop
        log_path = os.path.join(opts.save_dir, 'alog.txt')
        f = open(log_path, 'w')
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            start = time.time()
            train_epoch_sl(
                model,
                optimizer,
                lr_scheduler,
                epoch,
                train_dataset,
                val_datasets,
                problem,
                tb_logger,
                opts, f
            )
            end = time.time()
            print(f"training time cost {end-start}")
        f.close()


if __name__ == "__main__":
    run(get_options())
