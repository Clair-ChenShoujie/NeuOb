import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import hydra
import networkx as nx
import numpy as np
import torch as th
import torch.multiprocessing as mp
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import graph_generation as gg

def load_fold_data(output_dir, fold_idx):
    """
    Load data for a specific fold.

    Args:
        output_dir (str): Directory where fold data is saved.
        fold_idx (int): Fold index (1-based).

    Returns:
        dict: A dictionary containing train, val, and test data and labels.
    """
    fold_data = {}

    # Generate file paths
    train_graphs_path = os.path.join(output_dir, f"train_graphs_{fold_idx}.pkl")
    val_graphs_path = os.path.join(output_dir, f"val_graphs_{fold_idx}.pkl")
    test_graphs_path = os.path.join(output_dir, f"test_graphs_{fold_idx}.pkl")

    train_labels_path = os.path.join(output_dir, f"train_labels_{fold_idx}.pkl")
    val_labels_path = os.path.join(output_dir, f"val_labels_{fold_idx}.pkl")
    test_labels_path = os.path.join(output_dir, f"test_labels_{fold_idx}.pkl")

    # Load data
    with open(train_graphs_path, "rb") as f:
        fold_data["train_graphs"] = pickle.load(f)
    with open(val_graphs_path, "rb") as f:
        fold_data["val_graphs"] = pickle.load(f)
    with open(test_graphs_path, "rb") as f:
        fold_data["test_graphs"] = pickle.load(f)

    # Load labels
    with open(train_labels_path, "rb") as f:
        fold_data["train_labels"] = pickle.load(f)
    with open(val_labels_path, "rb") as f:
        fold_data["val_labels"] = pickle.load(f)
    with open(test_labels_path, "rb") as f:
        fold_data["test_labels"] = pickle.load(f)

    return fold_data


def get_one_shot_items(cfg: DictConfig, train_graphs, train_label, adj_brain=None):
    # Train Dataset
    train_dataset = gg.data.DenseGraphDataset(
        # adjs=[nx.to_numpy_array(G, dtype=bool) for G in train_graphs],
        adjs=[nx.to_numpy_array(G) for G in train_graphs],
        # adjs=train_graphs,
        train_label=train_label,
        adj_brains=adj_brain,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Model
    features = 2 if cfg.diffusion.name == "discrete" else 1
    # features = 3 if cfg.diffusion.name == "discrete" else 1   # lables
    model = gg.model.PPGN(
        in_features=features * (1 + cfg.diffusion.self_conditioning),
        out_features=features,
        emb_features=cfg.model.emb_features,
        hidden_features=cfg.model.hidden_features,
        ppgn_features=cfg.model.ppgn_features,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )

    # Diffusion
    if cfg.diffusion.name == "discrete":
        diffusion = gg.diffusion.dense.DiscreteGraphDiffusion(
            self_conditioning=cfg.diffusion.self_conditioning,
            num_steps=cfg.diffusion.num_steps,
        )
    elif cfg.diffusion.name == "edm":
        diffusion = gg.diffusion.dense.EDM(
            self_conditioning=cfg.diffusion.self_conditioning,
            num_steps=cfg.diffusion.num_steps,
        )
    else:
        raise ValueError(f"Unknown diffusion name: {cfg.diffusion.name}")

    # Method
    method = gg.method.OneShot(diffusion=diffusion)

    return {"train_dataloader": train_dataloader, "method": method, "model": model}

def adj_to_binary_graph_networkx(adj_matrix, threshold=0.6):
    """
    Convert adjacency matrix to a binary NetworkX graph where edges are
    set to 1 if their absolute value is greater than the threshold.

    Args:
        adj_matrix (np.ndarray): Adjacency matrix (can be weighted).
        threshold (float): Threshold for binarizing the adjacency matrix.

    Returns:
        nx.Graph: A binary NetworkX graph.
    """
    graphs = []
    for idx, sample in enumerate(adj_matrix):
        G_nx = nx.Graph()
        # 获取节点数量
        num_nodes = sample.shape[0]
        if sample.shape[0] != sample.shape[1]:
            raise ValueError(f"Adjacency matrix at index {idx} is not square: shape {sample.shape}")
        # 添加所有节点到图中
        G_nx.add_nodes_from(range(num_nodes))

        # 遍历上三角部分的邻接矩阵元素
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = sample[i, j]

                # 如果权重大于阈值或小于负阈值，则添加到图中，并将权重作为边的属性
                if weight >= threshold:
                    G_nx.add_edge(i, j, weight=1)  # 将权重直接添加到边的属性中
                    # G_nx.add_edge(i, j, weight=True)  # 将权重直接添加到边的属性中
                if weight <= -threshold:
                    G_nx.add_edge(i, j, weight=1)  # 将权重直接添加到边的属性中
                    # G_nx.add_edge(i, j, weight=2)  # 将权重直接添加到边的属性中
                    # G_nx.add_edge(i, j, weight=True)  # 将权重直接添加到边的属性中

        graphs.append(G_nx)

    return graphs

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.debugging:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Fix random seeds
    random.seed(0)
    np.random.seed(0)
    th.manual_seed(0)

    # Metrics
    validation_metrics = [
        gg.metrics.NodeDegree(),
        gg.metrics.ClusteringCoefficient(),
        gg.metrics.OrbitCount(),
        gg.metrics.Spectral(),
        gg.metrics.Wavelet(),
        gg.metrics.Uniqueness(),
        gg.metrics.Novelty(),
        gg.metrics.LCCDiff(),
        gg.metrics.CPLDiff(),
        gg.metrics.PLADiff(),
        gg.metrics.SCCDiff(),
        gg.metrics.REDEDiff(),
        gg.metrics.NSDiff(),
        gg.metrics.ccDiff(),
        gg.metrics.bcDiff(),
        gg.metrics.dcDiff(),
    ]

    i = int(cfg.training.kf_flod_index)
    # ./10kfold_data
    output_dir = "./5_kfold_data"
    fold_data = load_fold_data(output_dir, i)
    train_adj = fold_data["train_graphs"]
    train_label = fold_data["train_labels"]
    validation_adj = fold_data["val_graphs"]
    val_label = fold_data["val_labels"]
    test_adj = fold_data["test_graphs"]
    test_label = fold_data["test_labels"]

    # 转为0，1
    train_graphs = adj_to_binary_graph_networkx(train_adj)
    validation_graphs = adj_to_binary_graph_networkx(validation_adj)
    test_graphs = adj_to_binary_graph_networkx(test_adj)

    # Method
    # method_items = get_one_shot_items(cfg, train_adj, train_label)
    method_items = get_one_shot_items(cfg, train_graphs, train_label, train_adj)
    method_items = defaultdict(lambda: None, method_items)
    # Trainer
    th.set_float32_matmul_precision("high")
    trainer = gg.training.Trainer(
        sign_net=method_items["sign_net"],
        model=method_items["model"],  # ppgn
        method=method_items["method"],  # expansion
        train_dataloader=method_items["train_dataloader"],
        train_graphs=train_graphs,
        validation_graphs=validation_graphs,
        # test_graphs=test_graphs,
        test_graphs=train_graphs,
        train_adj=train_adj,
        validation_adj=validation_adj,
        # test_adj=test_adj,
        test_adj=train_adj,
        train_label=train_label,
        validation_label=val_label,
        # test_label=test_label,
        test_label=train_label,
        metrics=validation_metrics,
        cfg=cfg,
    )
    if cfg.testing:
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
