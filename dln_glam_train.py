import logging
import random
import time

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torch_geometric
import torch_geometric.nn.inits
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from GLAM.common import PageNodes, PageEdges
from GLAM.models import GLAMGraphNetwork
from dln_glam_prepare import DLNDataset, CLASSES_MAP


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Stopwatch:
    def __init__(self):
        self.elapsed = 0.

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start

    def __str__(self):
        return f"{self.elapsed:.4f} seconds"


def set_seed(n):
    torch.manual_seed(n)
    random.seed(n)
    np.random.seed(n)


def data_reset_dtype(data: torch_geometric.data.data.BaseData) -> torch_geometric.data.data.BaseData:
    data.node_features = data.node_features.to(torch.float32)
    data.edge_index = data.edge_index.to(torch.int64)
    data.edge_features = data.edge_features.to(torch.float32)
    data.node_probs = data.node_probs.to(torch.float32)
    data.edge_probs = data.edge_probs.to(torch.float32)
    return data


def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Create or load model
    model_filepath = "models/glam_dln.pt"
    set_seed(42)

    # if os.path.exists(model_filepath):
    #     model = glam.glam.GLAMGraphNetwork(PageNodes.features_len, PageEdges.features_len, 512, len(CLASSES_MAP))
    #     model.load_state_dict(torch.load(model_filepath))
    # else:
    model = GLAMGraphNetwork(PageNodes.features_len, PageEdges.features_len, 512, len(CLASSES_MAP))
    model = model.to(device)

    # TODO: normalize
    # transforms = torch_geometric.transforms.Compose([
    #     torch_geometric.transforms.NormalizeFeatures(attrs=['node_features', 'edge_features']),
    # ])

    train_dataset = DLNDataset("/home/i/dataset/DocLayNet/glam", 'train', transform=None, pre_transform=None)
    val_dataset = DLNDataset("/home/i/dataset/DocLayNet/glam", 'val', transform=None, pre_transform=None)
    # test_dataset = DLNDataset("/home/i/dataset/DocLayNet/glam", 'test', transform=None, pre_transform=None)

    train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
    val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)
    # test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Train parameters
    epochs = 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    edge_loss_scale = 4

    # Progress bars
    pbar_epoch = tqdm.tqdm(total=epochs, unit="epoch", position=0)
    pbar_train = tqdm.tqdm(total=train_loader.__len__(), unit="batch", position=1)
    pbar_val = tqdm.tqdm(total=val_loader.__len__(), unit="batch", position=2)

    # Train
    def closure() -> float:
        optimizer.zero_grad()

        node_class_scores, edge_class_scores = model(example)

        node_class_loss = F.cross_entropy(node_class_scores, example.node_probs)  # multi-class classification problem
        edge_class_loss = F.binary_cross_entropy_with_logits(edge_class_scores, example.edge_probs[..., None])  # multi-label classification problem
        loss = node_class_loss + edge_loss_scale * edge_class_loss
        loss.backward()
        return loss.item()

    with logging_redirect_tqdm():
        for epoch in range(epochs):
            # Train
            pbar_train.reset()
            model.train()
            for i, data in enumerate(train_loader.__iter__()):
                assert isinstance(data, torch_geometric.data.Data)
                example = data.clone()
                example = data_reset_dtype(example)
                example = example.to(device)

                loss = optimizer.step(closure)
                logger.info(f"Loss: {loss:.4f}")
                pbar_train.update(1)

            # Validation
            pbar_val.reset()
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader.__iter__()):
                    assert isinstance(data, torch_geometric.data.Data)
                    example = data.clone()
                    example = data_reset_dtype(example)
                    example = example.to(device)

                    node_class_scores, edge_class_scores = model(example)
                    node_class_loss = F.cross_entropy(node_class_scores, example.node_probs)  # multi-class classification problem
                    edge_class_loss = F.binary_cross_entropy_with_logits(edge_class_scores, example.edge_probs[..., None])  # multi-label classification problem
                    loss = node_class_loss + edge_loss_scale * edge_class_loss

                    edge_prob_threshold = 0.5
                    graph = nx.Graph()
                    for k in range(example.edge_index.shape[1]):
                        src_node_i = example.edge_index[0, k].item()
                        dst_node_i = example.edge_index[1, k].item()
                        edge_prob = example.edge_probs[k].item()

                        if edge_prob >= edge_prob_threshold:
                            graph.add_edge(src_node_i, dst_node_i, weight=edge_prob)
                        else:
                            graph.add_node(src_node_i)
                            graph.add_node(dst_node_i)

                    clusters: list[set[int]] = list(nx.connected_components(graph))
                    # cluster_min_spanning_boxes: list[Polygon] = [
                    #     Polygon([
                    #         # (min(nodes[node_i].bbox_min_x for node_i in cluster), min(nodes[node_i].bbox_min_y for node_i in cluster)),
                    #         # (max(nodes[node_i].bbox_max_x for node_i in cluster), min(nodes[node_i].bbox_min_y for node_i in cluster)),
                    #         # (max(nodes[node_i].bbox_max_x for node_i in cluster), max(nodes[node_i].bbox_max_y for node_i in cluster)),
                    #         # (min(nodes[node_i].bbox_min_x for node_i in cluster), max(nodes[node_i].bbox_max_y for node_i in cluster)),
                    #         (min(example.node_features[cluster, 2]), min(example.node_features[cluster, 3])),
                    #         (max(example.node_features[cluster, 4]), min(example.node_features[cluster, 3])),
                    #         (max(example.node_features[cluster, 4]), max(example.node_features[cluster, 5])),
                    #         (min(example.node_features[cluster, 2]), max(example.node_features[cluster, 5])),
                    #     ])
                    #     for cluster in clusters
                    # ]
                    cluster_classes: list[int] = torch.stack([example.node_probs[torch.tensor(list(cluster))].sum(dim=0) for cluster in clusters]).argmax(dim=1).tolist()

                    pbar_val.update(1)

            pbar_epoch.update(1)

        pbar_val.close()
        pbar_train.close()
        pbar_epoch.close()

    # Save model
    torch.save(model.state_dict(), model_filepath)


if __name__ == '__main__':
    main()
