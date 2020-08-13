import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.assume_mat = []
        self.edges = []
        self.g_list = []
        self.mid_feature = 0
        self.mid_edges = 0
        self.mid_g = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag = None):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    #{label0:0}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        #第一行是图的个数
        for i in range(n_g):
            #此循环下是每一个图
            row = f.readline().strip().split()
            # 便是每张图的数据有n行，其中第一行为节点数和label 后面 n-1行为结点属性
            n, l = [int(w) for w in row]
            #n是图中结点数
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            # 给feature编个号
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                # row[0]是feature_dict的key，row[1]是连接边的数量，之后是边
                tmp = int(row[1]) + 2
                # row[1]是一个节点所连接边的数量 2代表row[0] row[1]两个位置
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            # print(node_tags)
            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat
    for g in g_list:
        #g是S2VGraph的一个对象，g.g才是nx.Graph()的对象
        g.neighbors = [[] for i in range(len(g.g))]
        # []中一个[]代表一个结点的邻居
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        #形成[i,j],[j,i]的有向边

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        # g.edge_mat = torch.LongTensor(edges).transpose(0, 1)
        adj = nx.adj_matrix(g.g).tocoo()
        g.edge_mat = torch.from_numpy(np.mat([adj.row, adj.col], dtype=np.int64))

    if degree_as_tag:
        print("degree as feature")
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        # print(g.node_tags)
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}
    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        g.mid_feature = g.node_features
        # print(g.node_features.size())


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def separate_data(graph_list, train_set_ratio):

    num_train = int(len(graph_list) * train_set_ratio)
    num_val = int(len(graph_list) * ((1 - train_set_ratio) / 2))
    num_test = len(graph_list) - num_train - num_val

    idx_list = np.random.permutation(len(graph_list))
    train_idx, val_idx, test_idx = idx_list[:num_train], idx_list[num_train:num_train+num_val], idx_list[num_train+num_val:]
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]
    val_graph_list = [graph_list[i] for i in val_idx]

    return train_graph_list, val_graph_list, test_graph_list


# if __name__ == "__main__":
#     g_list, len_label_dict=load_data("MUTAG", False)
#     g_list = [1,2,3,4,5,6,7,8,9,10]
#     train, val, test = separate_data(g_list, 0.8)



