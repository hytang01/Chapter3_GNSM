import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, LayerNorm, ReLU, LeakyReLU, ELU
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn_graph, SAGEConv, to_hetero
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_geometric.data import Data,Dataset, InMemoryDataset, download_url, HeteroData
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


import os
import os.path as osp
import pickle
from pickle import dump,load
import numpy as np
import math, time
import pandas as pd
import copy

class MLP(nn.Module):
    def __init__(self,input_size,hidden_sizes,output_size,num_hidden_layers,activation_type,group_norm_choice):
        super(MLP,self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size = output_size
        self.num_hidden_layers  = num_hidden_layers
        self.activation_type = activation_type
        self.group_norm_choice = group_norm_choice
        # fc
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
        # act
        self.activation = nn.ModuleList()
        if self.activation_type == 'relu':
            self.activation_nn = nn.ReLU()
        elif self.activation_type == 'elu':
            self.activation_nn = nn.ELU()
        else:
            self.activation_nn = nn.LeakyReLU()
        self.activation.append(self.activation_nn)
        # gn
        self.group_norm_MLP = nn.ModuleList()
        self.group_norm_MLP.append(nn.GroupNorm(2, self.hidden_sizes[0]))
        for i in range(self.num_hidden_layers-1):
            self.fc.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]))
            self.group_norm_MLP.append(nn.GroupNorm(2, self.hidden_sizes[i+1]))
            self.activation.append(self.activation_nn)
        self.fc.append(nn.Linear(self.hidden_sizes[-1], self.output_size))
        
    def forward(self,x):
        for i in range(self.num_hidden_layers):
            x = self.fc[i](x)
            if self.group_norm_choice == 'MLP':
                x = self.group_norm_MLP[i](x)
            x = self.activation[i](x)
        x = self.fc[i+1](x)
        return x
# number_layer, general framework

# run processor one step only
class GraphProcessor(MessagePassing):
    def __init__(self, latent_size, hidden_sizes, num_hidden_layers,aggr_choice,message_choice,processor_choice,activation_type, group_norm_choice):
        super().__init__(aggr=aggr_choice,flow='source_to_target')
        self.latent_size = latent_size # the latent size for encoder 128
        self.hidden_sizes = hidden_sizes # the hidden dims for all networks (shared the same list of dims) [128,128,...,128]
        self.num_hidden_layers = num_hidden_layers # number of hidden layers for all networks (same) 2
        self.message_choice = message_choice
        self.processor_choice = processor_choice
        self.activation_type = activation_type
        self.group_norm_choice = group_norm_choice
        if self.message_choice == -1:
            self.mlp = MLP(2*self.latent_size,self.hidden_sizes,self.latent_size,self.num_hidden_layers,self.activation_type, self.group_norm_choice) # input_size based on how code msg
        elif self.message_choice == 0 or self.message_choice == 1:
            self.mlp = MLP(3*self.latent_size,self.hidden_sizes,self.latent_size,self.num_hidden_layers,self.activation_type, self.group_norm_choice) # input_size based on how code msg
        else:
            self.mlp = MLP(4*self.latent_size,self.hidden_sizes,self.latent_size,self.num_hidden_layers,self.activation_type, self.group_norm_choice) # input_size based on how code msg
        self.node_mlp = MLP(2*self.latent_size,self.hidden_sizes,self.latent_size,self.num_hidden_layers,self.activation_type, self.group_norm_choice)
        self.node_mlp_less = MLP(self.latent_size,self.hidden_sizes,self.latent_size,self.num_hidden_layers,self.activation_type, self.group_norm_choice)
        
    def forward(self, x, edge_index, edge_attr): 
        out = self.propagate(edge_index,x=x,edge_attr=edge_attr)
        if self.processor_choice == 0:
            return out
        elif self.processor_choice == 1:
            out1 = self.node_mlp_less(out)
            return out1
        else:
            in_node_mlp = torch.cat((out,x),-1)
            out2 = self.node_mlp(in_node_mlp)
            return out2
    
    def message(self, x_i, x_j,edge_attr):
        if self.message_choice == -1:
            x_new = torch.cat((x_i,x_j),-1)
        elif self.message_choice == 0:
            x_new = torch.cat((x_i,x_j,edge_attr),-1)
        elif self.message_choice == 1:
            x_new = torch.cat((x_i,x_j-x_i,edge_attr),-1)
        else:
            x_new = torch.cat((x_i,x_j,x_j-x_i,edge_attr),-1)
        out = self.mlp(x_new)
        return out

# the main class for our surrogate model
class SIMGNN(nn.Module):
    def __init__(self, input_size,edge_input_size, output_size, latent_size, hidden_sizes, num_hidden_layers, num_message_passing_steps,aggr_choice,message_choice,is_res_net=False,processor_choice = 1,activation_type='relu',group_norm_choice='None'):
        super(SIMGNN,self).__init__()
        self.input_size = input_size # the input dim of features 8*
        self.edge_input_size = edge_input_size # the input dim of features 4*
        self.output_size = output_size # the output dim of labels 2*
        self.latent_size = latent_size # the latent size for encoder 128
        self.hidden_sizes = hidden_sizes # the hidden dims for all networks (shared the same list of dims) [128,128,...,128]
        self.num_hidden_layers = num_hidden_layers # number of hidden layers for all networks (same) 2
        self.num_message_passing_steps = num_message_passing_steps # number of GraphProcessor get called 10*
        self.aggr_choice=aggr_choice
        self.message_choice=message_choice
        self.is_res_net=is_res_net
        self.processor_choice=processor_choice
        self.activation_type=activation_type
        self.group_norm_choice=group_norm_choice
        self.group_norm_process = nn.GroupNorm(2, self.latent_size)
        self.encode_mlp = MLP(self.input_size,self.hidden_sizes,self.latent_size,self.num_hidden_layers,self.activation_type,self.group_norm_choice)
        self.encode_edge_mlp = MLP(self.edge_input_size,self.hidden_sizes,self.latent_size,self.num_hidden_layers,self.activation_type,self.group_norm_choice)
        self.decode_mlp = MLP(self.latent_size,self.hidden_sizes,self.output_size,self.num_hidden_layers,self.activation_type,self.group_norm_choice)
        self.process_one_step = GraphProcessor(self.latent_size,self.hidden_sizes,self.num_hidden_layers,self.aggr_choice,self.message_choice,self.processor_choice,self.activation_type,self.group_norm_choice)
        
    def forward(self,input_graph): 
        # S1. encoder - apply mlp to each node/edge for dimension reduction (7->128)
        encoded_features = self.encoder(input_graph)
        if self.message_choice != -1:
            encoded_edge_features = self.edge_encoder(input_graph)
        else:
            encoded_edge_features = []
        
        # S2. processor - apply several mlps in GNN style (128->->...->128)
        processed_features = self.processor(encoded_features,encoded_edge_features,input_graph)
    
        # S3. decoder - apply mlp to each node to recover the original dimension (128->2)
        decoded_features = self.decoder(processed_features,input_graph)
        
        return decoded_features
    
    def encoder(self,input_graph): # only node features considered currently
        input_features = input_graph['x']
        encoded_features = self.encode_mlp(input_features)
        return encoded_features
    
    def edge_encoder(self,input_graph): # only node features considered currently
        input_features = input_graph['edge_attr']
        encoded_features = self.encode_edge_mlp(input_features)
        return encoded_features
    
    def processor(self,encoded_features,encoded_edge_features,input_graph):
        processed_graph = input_graph
        x = encoded_features
        edge_attr = encoded_edge_features
        edge_index =  processed_graph.edge_index
        for i in range(self.num_message_passing_steps):  
            if self.is_res_net:
                x = self.process_one_step(x,edge_index,edge_attr) + x
                # group normalization
                if self.group_norm_choice == 'Processor':
                    x = self.group_norm_process(x)
            else:
                x = self.process_one_step(x,edge_index,edge_attr)
                if self.group_norm_choice == 'Processor':
                    x = self.group_norm_process(x)
        return x
    
    def decoder(self,processed_features,input_graph):
        decoded_graph = input_graph
        decoded_features = self.decode_mlp(processed_features)
        return decoded_features