import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy.sparse as sp
import torch
import time
import logging
from copy import deepcopy
# 找出一个图中节点数为k的所有子图
def ESU(graph, k):
    resault = []
    def extendSubgraph(v_sub, v_ext, v):
        # print('v_ext', v_ext)
        if len(v_sub) == k:
            sub = nx.subgraph(graph, v_sub)
            resault.append(sub)
            return
        while len(v_ext) != 0:
            w = v_ext.pop()
            ex_set = set()
            n_sub = set()
            for i in v_sub:
                n_sub = n_sub.union(set(nx.neighbors(graph, i)))
            n_w = set(nx.neighbors(graph, w))
            n_sub = v_sub.union(n_sub)
            n_w = n_w.difference(n_sub)
            jv_set = set()
            for j in n_w:
                if j <= v:
                    jv_set.add(j)
            n_w = n_w.difference(jv_set)
            v_newext = v_ext.union(n_w)
            v_newsub = v_sub.union(set({w}))
            # print("v_sub:{}".format(v_sub), "v_ext: {}".format(v_ext))
            extendSubgraph(v_newsub, v_newext, v)

    for vertex in graph.nodes:
        v_exten = set()
        # print('node', vertex)
        for nei_node in nx.neighbors(graph, vertex):
            if nei_node > vertex:
                v_exten.add(nei_node)

        extendSubgraph(set({vertex}), v_exten, vertex)

    return resault

def def_motif(motif):
    if motif == "triangle":
        triangle = nx.Graph()
        for i in range(1, 4):
            triangle.add_node(i)
        triangle.add_edge(1, 3)
        triangle.add_edge(2, 3)
        triangle.add_edge(1, 2)
        return triangle

    if motif == "3star":
        four = nx.Graph()
        for i in range(1, 5):
            four.add_node(i)
        four.add_edge(1, 3)
        four.add_edge(1, 4)
        four.add_edge(1, 2)
        return four

    if motif == "square":
        four = nx.Graph()
        for i in range(1, 5):
            four.add_node(i)
        four.add_edge(2, 3)
        four.add_edge(3, 4)
        four.add_edge(1, 4)
        four.add_edge(1, 2)
        return four

    if motif == "4-clique":
        four_clique = nx.Graph()
        for i in range(1, 5):
            four_clique.add_node(i)
        four_clique.add_edge(2, 3)
        four_clique.add_edge(3, 4)
        four_clique.add_edge(1, 4)
        four_clique.add_edge(1, 2)
        four_clique.add_edge(1, 3)
        four_clique.add_edge(2, 4)
        return four_clique


def match_motif_with_overlap(subgraphs, motif):
    start_time = time.time()
    # classes是所有的motif
    classes = []
    subgraph_matched = []

    for i in subgraphs:
        if nx.is_isomorphic(i, motif):
            subgraph_matched.append(i)
            classes.append(list(i.nodes))
    end_time = time.time()
    logging.warning("match time {}".format(end_time - start_time))
    print("number of subgraph {}".format(len(subgraphs)))
    print("num of matched subgraphs {}".format(len(subgraph_matched)))
    # motif_nodes is a set of nodes which are nodes of matched subgraphs
    motif_nodes = set()
    start_time = time.time()
    for i in range(len(subgraph_matched)-1, 0, -1):
        for j in range(i-1, -1, -1):
            overlap_nodes = set(classes[j]) & set(classes[i])
            if len(overlap_nodes) >= (motif.number_of_nodes()/2):
                classes.remove(classes[i])
                break
    for i in classes:
        motif_nodes = motif_nodes.union(set(i))
    end_time = time.time()
    logging.warning("wipe out overlap time {}".format(end_time-start_time))
    # print('motif_ndes', motif_nodes)
    return classes, motif_nodes

# def pool_without_overlap(g, motif_clique):
# # def match_motif_with_overlap(g, motif_clique):
#     g = g.deepcopy()
#     g = g.to_undirected()
#     start_time = time.time()
#     all_cliques = nx.algorithms.clique.enumerate_all_cliques(g)
#     classes = []
#     subgraph_matched = []
#     for clique in all_cliques:
#         if len(clique) == motif_clique.number_of_nodes():
#             subgraph_matched.append(clique)
#             classes.append(list(clique.nodes))
#     end_time = time.time()
#     logging.warning("match time {}".format(end_time - start_time))
#
#     start_time = time.time()
#     for i in range(len(subgraph_matched)-1, 0, -1):
#         for j in range(i-1, -1, -1):
#             overlap_nodes = set(classes[j]) & set(classes[i])
#             if len(overlap_nodes) >= (motif_clique.number_of_nodes()/2):
#                 classes.remove(classes[i])
#                 break
#     for i in classes:
#         motif_nodes = motif_nodes.union(set(i))
#     end_time = time.time()
#     logging.warning("wipe out overlap time {}".format(end_time-start_time))
#     # print('motif_ndes', motif_nodes)
#     return classes, motif_nodes






"""根据匹配的motif进行pooling操作，生成新的图"""
def gen_new_graph(classes, motif_nodes, motif, g):
    start_time = time.time()
    nodes_num_of_graph = nx.number_of_nodes(g)
    motif_size = len(nx.nodes(motif))
    none_motif_nodes = g.nodes - motif_nodes
    new_graph = nx.Graph()
    # node_dic key: original node in g, value: the corresponding node in the new graph
    node_dic = {}
    for node in g.nodes:
        node_dic[node] = []
    assum_list = []
    for i in range(len(classes)):
        new_graph.add_node(i)
        # temp_list is a row of assum_list and the assum_mat
        temp_list = [0]*nodes_num_of_graph
        for j in range(len(classes[i])):
            node_dic[classes[i][j]].append(i)
            # the feature of super node is the average or sum of pooling nodes
            temp_list[classes[i][j]] = 1/len(classes[i])
            # temp_list[classes[i][j]] = 1
        assum_list.append(temp_list)
    key_index = len(classes)
    for node in none_motif_nodes:
        node_dic[node].append(key_index)
        new_graph.add_node(key_index)
        key_index += 1
        temp_list = [0] * nodes_num_of_graph
        try:
            temp_list[node] = 1
        except:
            print('node', node)
            print('nodes_num_of_graph', len(temp_list))
            graph_nodes = list(g.nodes)
            graph_nodes.sort()
            print('graph', graph_nodes)
            nx.draw_networkx(g, with_labels=True)
            plt.show()
            break
        assum_list.append(temp_list)

    for edge in nx.edges(g):
        for i in node_dic[edge[0]]:
            for j in node_dic[edge[1]]:
                if i == j:
                    continue
                else:
                    new_graph.add_edge(i, j)
    assum_mat = torch.from_numpy(np.array(assum_list, dtype=np.float32))
    adj = nx.adj_matrix(new_graph).tocoo()

    end_time = time.time()
    logging.warning("gen new graph time {}".format(end_time-start_time))
    return assum_mat, adj, new_graph

def edge_to_nxGraph(edges, num_nodes):
    g = nx.Graph()
    # print("data device", edges.device)
    if edges.device != "cpu":
        # print("transform")
        edges = edges.cpu()
    edge_list = edges.t().numpy()
    g.add_edges_from(edge_list)
    ori_nodes = list(g.nodes)
    ori_nodes.sort()
    # if a node is separated, this node's index will not occur in the edge_index
    if g.number_of_nodes() != num_nodes:
        nodes_list = [i for i in range(num_nodes)]
        single_nodes = list(set(nodes_list) - set(ori_nodes))
        g.add_nodes_from(single_nodes)
    return g




