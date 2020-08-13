import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import signal

from util import load_data, separate_data
from models.model import Motif_Pool
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from match import *

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = len(train_graphs)//args.batch_size
    loss_accum = 0
    train_all_index = np.random.permutation(len(train_graphs))
    index=[]
    for iter in range(len(train_graphs)//args.batch_size):
        selected_idx = train_all_index[iter*args.batch_size:(iter+1)*args.batch_size]
        for i in selected_idx:
            if i not in index:
                index.append(i)
            else:
                print("idx", selected_idx)

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)
        # print("output shape", output.shape)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        # print("label type", labels.shape)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss


    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        model_output = model([graphs[j] for j in sampled_idx]).detach()
        # print("model_output_shape",model_output.shape)
        output.append(model_output)
    return torch.cat(output, 0)

def evaluate(args, model, device, graphs, name):
    model.eval()

    output = pass_data_iteratively(model, graphs, args.batch_size)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc = correct / float(len(graphs))

    val_loss = 0
    if name == "val":
        val_loss = criterion(output, labels)
        print(name, "val_loss : %f" % val_loss.item())

    print(name, "accuracy : %f " % acc)

    return acc, val_loss

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="REDDITBINARY",
                        help='name of dataset (default: PTC)')
    parser.add_argument('--motif', type=str, default="triangle",
                        help='triangle, 3star, square')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='number of hidden units (default: 32)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,help='dropout ratio')
    parser.add_argument('--degree_as_tag', action="store_true",default=False,
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type=str, default="", help='output file')
    parser.add_argument("--train_set_ratio", type=float, default=0.8, help='the ratio of training set')
    parser.add_argument("--adaptive_ratio", type=float, default=0.5, help='the ratio of neighbors in adaptive graph')
    parser.add_argument('--num_pool_layers', type=int, default=3, help='the number of pooling layers')
    parser.add_argument('--neighbor_pool_type', type=str, default='average', choices=['sum', 'average', 'max'],
                        help='pooling for neighboring nodes')
    args = parser.parse_args()
    print("degree as tag", args.degree_as_tag)

    args.device = 'cpu'
    np.random.seed(0)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        args.device = 'cuda:4'

    graphs, args.num_classes = load_data(args.dataset, args.degree_as_tag)

    # motif = def_motif(args.motif)
    # motif_size = motif.number_of_nodes()
    #
    # four_clique = def_motif("4-clique")
    # four_clique_size = four_clique.number_of_nodes()
    clique_list = [5,4,3]
    # reverse graphs[] to make sure the index of graph is right when remove a graph from graphs[]
    # for graph in graphs[::-1]:
    #     print("original graph",  graph.g.number_of_nodes())
    #     g = graph.g
    #     nx.draw_networkx(g, with_labels=True)
    #     plt.show()
    #     for pool_layer in range(args.num_pool_layers):
    #         classes, motif_nodes = pool_without_overlap(g, clique_list[pool_layer])
    #         assum_mat, adj, g = gen_new_graph(classes, motif_nodes, g)
    #
    #         print("new graph", g.number_of_nodes())
    #         nx.draw_networkx(g, with_labels=True)
    #         plt.show()
    #         assum_mat = assum_mat.to(args.device)
    #         edges = torch.from_numpy(np.mat([adj.row, adj.col], dtype=np.int64))
    #         graph.g_list.append(g)
    #         graph.assume_mat.append(assum_mat)
    #         graph.edges.append(edges)



    print("len graphs", len(graphs))
    train_graphs, val_graphs, test_graphs = separate_data(graphs, args.train_set_ratio)
    # print("train", len(train_graphs), len(test_graphs))
    print("input_dim", train_graphs[0].node_features.shape[1])
    input_dim = train_graphs[0].node_features.shape[1]
    model = Motif_Pool(input_dim, args.hidden_dim, args).to(args.device)
    print('# model parameters:', sum(param.numel() for param in model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    writer = SummaryWriter('log')

    loss_list = []
    train_acc_list = []
    accuracy_list = []
    val_loss_list = []
    x1 = []
    x2 = []
    y1 = loss_list
    y2 = accuracy_list
    min_loss = 1e10

    for epoch in range(1, args.epochs + 1):
        print("epoch:{}".format(epoch))

        avg_loss = train(args, model, args.device, train_graphs, optimizer, epoch)
        acc_train, _ = evaluate(args, model, args.device, train_graphs, "train")
        acc_val, val_loss = evaluate(args, model, args.device, val_graphs, "val")
        if val_loss < min_loss:
            torch.save(model.state_dict(), 'latest.pth')
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss

        scheduler.step()

        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/Accu', acc_train, epoch)
        writer.add_scalar('Val/Accu', acc_val, epoch)

        x1.append(epoch)
        loss_list.append(avg_loss)
        train_acc_list.append(acc_train)
        x2.append(epoch)
        accuracy_list.append(acc_val)
        val_loss_list.append(val_loss)

        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_val))
                f.write("\n")
        print("")

        # print('model.eps', model.eps)
    writer.close()
    model.load_state_dict(torch.load('latest.pth'))
    test_acc, _ = evaluate(args, model, args.device, test_graphs, "test")
    print("Test accuracy:{}".format(test_acc))

    fig = plt.figure(num=1, figsize=(15, 8), dpi=80)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_title('train')
    ax2.set_title('validation')
    # ax1.set_ylim(0, 1)
    # ax2.set_ylim(0, 1)
    plot1 = ax1.plot(x1, y1, marker='.', color='g', label='train_loss')
    plot3 = ax1.plot(x1, train_acc_list, marker='.', color='r', label='train_acc')
    plot2 = ax2.plot(x2, y2, marker='.', color='r', label='val_acc')
    plot4 = ax2.plot(x2, val_loss_list, marker='.', color='g', label='val_loss')
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    main()
