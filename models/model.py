import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import sys
# sys.path.append("models/")
from models.mlp import MLP
from torch_geometric.nn import GCNConv,GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from match import *

class Motif_Pool(nn.Module):
    def __init__(self, in_features, hidden_dim, args):
        super(Motif_Pool, self).__init__()

        self.device = args.device
        self.num_pool_layers = args.num_pool_layers
        self.dropout_ratio = args.dropout_ratio
        self.args = args


        self.linear = torch.nn.Linear(hidden_dim*2, args.num_classes)
        self.mlps = torch.nn.ModuleList()
        self.gins = torch.nn.ModuleList()
        self.gcns = torch.nn.ModuleList()

        self.linears_prediction = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = nn.ModuleList()
        self.pre_aggregation = nn.ModuleList()
        self.pre_aggregation.append(GCNConv(in_features, hidden_dim))
        self.pre_aggregation.append(GCNConv(hidden_dim, hidden_dim))
        self.pre_bn1 = nn.BatchNorm1d(hidden_dim)
        self.pre_bn2 = nn.BatchNorm1d(hidden_dim)



        for pool_layer in range(self.num_pool_layers+1):
            # self.linears_prediction.append(nn.Linear(hidden_dim*2, args.num_classes))
            if pool_layer == 0:
            #     self.mlps.append(MLP(in_features, hidden_dim, hidden_dim))
            #     self.gins.append(GINConv(self.mlps[pool_layer*2], 0, False))
            #     self.mlps.append(MLP(hidden_dim, hidden_dim, hidden_dim))
            #     self.gins.append(GINConv(self.mlps[pool_layer*2+1], 0, False))
                self.gcns.append(GCNConv(in_features, hidden_dim))
            #     self.gcns.append(GCNConv(hidden_dim, hidden_dim))
            # else:
            self.mlps.append(MLP(hidden_dim, hidden_dim, hidden_dim))
            self.gins.append(GINConv(self.mlps[pool_layer*2], 0, False))
            self.mlps.append(MLP(hidden_dim, hidden_dim, hidden_dim))
            self.gins.append(GINConv(self.mlps[pool_layer*2+1], 0, False))
            self.gcns.append(GCNConv(hidden_dim, hidden_dim))
            # self.gcns.append(GCNConv(hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            # self.batch_norms.append(nn.BatchNorm1d(hidden_dim))


    def __preprocess(self, batch_graph, pool_layer, start_idx):
        ###create

        edge_mat_list = []
        node_feature_list = []
        # start_idx = [0]
        for i, graph in enumerate(batch_graph):
            # make sure that the index of edges in the batch are continuous
            edge_mat_list.append(graph.edges[pool_layer] + start_idx[i])
            try:
                graph.mid_feature = torch.matmul(graph.assume_mat[pool_layer], graph.mid_feature)
            except Exception as e:
                print(e)
            node_feature_list.append(graph.mid_feature)
            #graph.edge_mat + start_idx[i] 每个元素加上start_idx[i]
        edge_index = torch.cat(edge_mat_list, 1)
        x = torch.cat(node_feature_list, 0)

        return edge_index.to(self.device), x.to(self.device)


    def forward(self, batch_graph):
        hidden_rep = []
        start_clique = 5
        clique_list = [5, 4, 3]
        for i, graph in enumerate(batch_graph):
            graph.mid_feature = graph.node_features
            graph.mid_edges = graph.edge_mat
            graph.mid_g = graph.g

        for pool_layer in range(self.args.num_pool_layers+1):
            edge_mat_list = []
            node_feature_list = []
            start_idx = [0]
            # print("pool layer", pool_layer)
            for i, graph in enumerate(batch_graph):
                new_mat = new_matrix(graph.mid_feature, self.args.adaptive_ratio)
                adj = nx.adj_matrix(graph.mid_g).todense() + new_mat.numpy()
                graph.mid_g = nx.from_numpy_matrix(adj)
                adj = nx.adj_matrix(graph.mid_g).tocoo()
                graph.mid_edges = torch.from_numpy(np.mat([adj.row, adj.col], dtype=np.int64))
                edge_mat_list.append(graph.mid_edges + start_idx[i])
                node_feature_list.append(graph.mid_feature)
                start_idx.append(start_idx[i] + len(graph.mid_g))
            # print("node feature list", node_feature_list)
            edge_index = torch.cat(edge_mat_list, 1).to(self.device)
            x = torch.cat(node_feature_list, 0).to(self.device)
            x = F.relu(self.batch_norms[pool_layer](self.gcns[pool_layer](x, edge_index)))
            # x = F.relu(self.pre_bn1(self.pre_aggregation[0](x, edge_index)))
            # x = F.relu(self.pre_bn2(self.pre_aggregation[1](x, edge_index)))
            if pool_layer == self.args.num_pool_layers:
                break

            batch = []
            for i, graph in enumerate(batch_graph):
                graph.mid_feature = x[start_idx[i]:start_idx[i + 1]]
                classes, motif_nodes = pool_without_overlap(graph.mid_g, clique_list[pool_layer])
                assum_mat, adj, graph.mid_g = gen_new_graph(classes, motif_nodes, graph.mid_g)
                assum_mat = assum_mat.to(self.device)
                graph.mid_feature = torch.matmul(assum_mat, graph.mid_feature)
                batch[start_idx[i]:start_idx[i + 1]] = [i] * (start_idx[i + 1] - start_idx[i])

            batch = torch.LongTensor(batch).to(self.device)
            hidden_rep.append(torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1))

        # for i, graph in enumerate(batch_graph):
        #     graph.mid_feature = x[start_idx[i]:start_idx[i + 1]]
        #
        # for pool_layer in range(self.num_pool_layers):
        #     start_idx = [0]
        #     for i, graph in enumerate(batch_graph):
        #         start_idx.append(start_idx[i] + len(graph.g_list[pool_layer]))
        #     edge_index, x = self.__preprocess(batch_graph, pool_layer, start_idx)
        #
        #     # x = F.relu(self.batch_norms[pool_layer](self.gcns[pool_layer](x, edge_index)))
        #     x = F.relu(self.batch_norms[pool_layer*2](self.gcns[pool_layer*2](x, edge_index)))
        #     x = F.relu(self.batch_norms[pool_layer * 2+1](self.gcns[pool_layer * 2+1](x, edge_index)))
        #
        #     # x = F.relu(self.batch_norms[pool_layer](self.gcns[pool_layer](x, edge_index)))
        #     # x = F.dropout(x, self.dropout_ratio, training=self.training)
        #     if pool_layer < self.num_pool_layers-1:
        #         for i, graph in enumerate(batch_graph):
        #             graph.mid_feature = x[start_idx[i]:start_idx[i+1]]
        #
        #
        #     batch = []
        #     for i, graph in enumerate(batch_graph):
        #         batch[start_idx[i]:start_idx[i+1]] = [i]*(start_idx[i+1]-start_idx[i])
        #     batch = torch.LongTensor(batch).to(self.device)
        #     hidden_rep.append(torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1))

        h = 0
        for layer, rep in enumerate(hidden_rep):
            h += F.dropout(rep, self.dropout_ratio, training=self.training)

        # h =  F.dropout(hidden_rep[-1], self.dropout_ratio, training=self.training)
        # h = hidden_rep[-1]

        h = self.linear(h)
        # h = F.dropout(h, self.dropout_ratio, training=self.training)
        return h
