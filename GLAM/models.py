import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TAGConv
from torch_geometric.data import Data


class NodeEncoder(nn.Module):
    def __init__(self, node_features_len: int, initial_hidden_len: int, activation=nn.GELU()):
        super(NodeEncoder, self).__init__()
        self.batch_norm = nn.BatchNorm1d(node_features_len)
        self.activation = activation
        self.linear1 = nn.Linear(node_features_len, (cur_len := initial_hidden_len))
        self.conv1 = TAGConv(cur_len, (cur_len := cur_len // 2))
        self.linear2 = nn.Linear(cur_len, (cur_len := cur_len // 2))
        self.conv2 = TAGConv(cur_len, (cur_len := cur_len // 2))

        self.linear3 = nn.Linear((cur_len := cur_len + node_features_len), (cur_len := cur_len // 2))
        self.linear4 = nn.Linear(cur_len, (cur_len := cur_len // 2))
        self.embeddings_len = cur_len

    def forward(self, data: Data) -> torch.Tensor:
        x = data.node_features
        x = self.batch_norm(x)
        x = self.activation(self.linear1(x))
        x = self.activation(self.conv1(x, data.edge_index))
        x = self.activation(self.linear2(x))
        x = self.activation(self.conv2(x, data.edge_index))
        x = torch.cat([x, data.node_features], dim=1)
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x


class NodeClassifier(nn.Module):
    def __init__(self, node_embeddings_len: int, classes_len: int, activation=nn.GELU()):
        super(NodeClassifier, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(node_embeddings_len, classes_len)
        self.classes_len = classes_len

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(node_embeddings))


class EdgeEncoder(nn.Module):
    def __init__(self, node_embeddings_len: int, edge_feature_len: int, activation=nn.GELU()):
        super(EdgeEncoder, self).__init__()
        cur_len = node_embeddings_len + edge_feature_len
        self.batch_norm = nn.BatchNorm1d(cur_len)
        self.activation = activation
        self.linear1 = nn.Linear(cur_len, (cur_len := cur_len // 2))
        self.linear2 = nn.Linear(cur_len, (cur_len := cur_len // 2))
        self.embeddings_len = cur_len

    def forward(self, node_embeddings: torch.Tensor, data: Data) -> torch.Tensor:
        aggregated_node_features = (node_embeddings[data.edge_index[0]] + node_embeddings[data.edge_index[1]]) / 2
        x = torch.cat([aggregated_node_features, data.edge_features], dim=1)
        x = self.batch_norm(x)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x


class EdgeClassifier(nn.Module):
    def __init__(self, edge_embeddings_len: int, activation=nn.GELU()):
        super(EdgeClassifier, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(edge_embeddings_len, (cur_len := 1))
        self.classes_len = cur_len

    def forward(self, edge_embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(edge_embeddings))


class GLAMGraphNetwork(nn.Module):
    """https://arxiv.org/abs/2308.02051"""

    def __init__(self, node_features_len, edge_feature_len, initial_hidden_len, node_classes_len):
        super(GLAMGraphNetwork, self).__init__()
        self.node_encoder = NodeEncoder(node_features_len=node_features_len, initial_hidden_len=initial_hidden_len)
        self.node_classifier = NodeClassifier(node_embeddings_len=self.node_encoder.embeddings_len, classes_len=node_classes_len)
        self.edge_encoder = EdgeEncoder(node_embeddings_len=self.node_encoder.embeddings_len, edge_feature_len=edge_feature_len)
        self.edge_classifier = EdgeClassifier(edge_embeddings_len=self.edge_encoder.embeddings_len)

    def forward(self, data: Data) -> (torch.Tensor, torch.Tensor):
        node_embeddings = self.node_encoder(data)
        node_class_scores = self.node_classifier(node_embeddings)
        edge_embeddings = self.edge_encoder(node_embeddings, data)
        edge_class_scores = self.edge_classifier(edge_embeddings)
        return node_class_scores, edge_class_scores


def main():
    model = GLAMGraphNetwork(10, 20, 30, 40)
    print(model)


if __name__ == '__main__':
    main()
