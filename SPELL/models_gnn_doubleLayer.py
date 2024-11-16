import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, SAGEConv, EdgeConv
from torch_geometric.utils.dropout import dropout_adj
from unet_parts import *
from unet_model import UNet

class SPELL(torch.nn.Module):
    def __init__(self, channels, feature_dim=1024, dropout=0, dropout_a=0, da_true=False, proj_dim=64):
        self.channels = channels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.dropout_a = dropout_a
        self.da_true = da_true
        super(SPELL, self).__init__()

        self.layerspf = nn.Linear(4, proj_dim) # projection layer for spatial features (4 -> 64)
        self.layer011 = nn.Linear(self.feature_dim//2+proj_dim, self.channels[0])
        self.layer012 = nn.Linear(self.feature_dim//2, self.channels[0])
        self.batch01 = BatchNorm(self.channels[0])
        # print("nn.Linear : ",nn.Linear(2*self.channels[0], self.channels[0]))

        self.layer11 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.batch11 = BatchNorm(self.channels[0])
        self.layerOther = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 2*self.channels[0]), nn.ReLU(), nn.Linear(2*self.channels[0], self.channels[0])))
        self.layerOther2 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], 4*self.channels[0]), nn.ReLU(), nn.Linear(4*self.channels[0], self.channels[0])))

        self.layer12 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.batch12 = BatchNorm(self.channels[0])
        self.layer13 = EdgeConv(nn.Sequential(nn.Linear(2*self.channels[0], self.channels[0]), nn.ReLU(), nn.Linear(self.channels[0], self.channels[0])))
        self.batch13 = BatchNorm(self.channels[0])

        self.layer21 = SAGEConv(self.channels[0], self.channels[1])
        self.batch21 = BatchNorm(self.channels[1])

        self.layer31 = SAGEConv(self.channels[1], 1)
        self.layer32 = SAGEConv(self.channels[1], 1)
        self.layer33 = SAGEConv(self.channels[1], 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr



        spf = x[:, self.feature_dim:self.feature_dim+4] # coordinates for the spatial features (dim: 4)
        edge_index1 = edge_index[:, edge_attr>=0]
        edge_index2 = edge_index[:, edge_attr<=0]

        x_visual = self.layer011(torch.cat((x[:,self.feature_dim//2:self.feature_dim], self.layerspf(spf)), dim=1))
        x_audio = self.layer012(x[:,:self.feature_dim//2])
        x = x_audio + x_visual
        # print(x.shape)
        x = self.batch01(x)
        x = F.relu(x)

        edge_index1m, _ = dropout_adj(edge_index=edge_index1, p=self.dropout_a, training=self.training if not self.da_true else True)


        x1 = self.layer11(x, edge_index1m)
        x1 = self.batch11(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1 = self.layer21(x1, edge_index1)
        x1 = self.batch21(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x11 = self.layerOther(x, edge_index1m)
        x11 = self.batch11(x11)
        x11 = F.relu(x11)
        x11 = F.dropout(x11, p=self.dropout, training=self.training)
        x11 = self.layer21(x11, edge_index1)
        x11 = self.batch21(x11)
        x11 = F.relu(x11)
        x11 = F.dropout(x11, p=self.dropout, training=self.training)

        x111 = self.layerOther2(x, edge_index1m)
        x111 = self.batch11(x111)
        x111 = F.relu(x111)
        x111 = F.dropout(x111, p=self.dropout, training=self.training)
        x111 = self.layer21(x111, edge_index1)
        x111 = self.batch21(x111)
        x111 = F.relu(x111)
        x111 = F.dropout(x111, p=self.dropout, training=self.training)



        edge_index2m, _ = dropout_adj(edge_index=edge_index2, p=self.dropout_a, training=self.training if not self.da_true else True)
        x2 = self.layer12(x, edge_index2m)
        x2 = self.batch12(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = self.layer21(x2, edge_index2)
        x2 = self.batch21(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x22 = self.layerOther(x, edge_index2m)
        x22 = self.batch12(x22)
        x22 = F.relu(x22)
        x22 = F.dropout(x22, p=self.dropout, training=self.training)
        x22 = self.layer21(x22, edge_index1)
        x22 = self.batch21(x22)
        x22 = F.relu(x22)
        x22 = F.dropout(x22, p=self.dropout, training=self.training)

        x222 = self.layerOther2(x, edge_index2m)
        x222 = self.batch12(x222)
        x222 = F.relu(x222)
        x222 = F.dropout(x222, p=self.dropout, training=self.training)
        x222 = self.layer21(x222, edge_index1)
        x222 = self.batch21(x222)
        x222 = F.relu(x222)
        x222 = F.dropout(x222, p=self.dropout, training=self.training)


        # Undirected graph
        edge_index3m, _ = dropout_adj(edge_index=edge_index, p=self.dropout_a, training=self.training if not self.da_true else True)
        x3 = self.layer13(x, edge_index3m)
        x3 = self.batch13(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        x3 = self.layer21(x3, edge_index)
        x3 = self.batch21(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        x33 = self.layerOther(x, edge_index3m)
        x33 = self.batch13(x33)
        x33 = F.relu(x33)
        x33 = F.dropout(x33, p=self.dropout, training=self.training)
        x33 = self.layer21(x33, edge_index1)
        x33 = self.batch21(x33)
        x33 = F.relu(x33)
        x33 = F.dropout(x33, p=self.dropout, training=self.training)

        x333 = self.layerOther2(x, edge_index3m)
        x333 = self.batch13(x333)
        x333 = F.relu(x333)
        x333 = F.dropout(x333, p=self.dropout, training=self.training)
        x333 = self.layer21(x333, edge_index1)
        x333 = self.batch21(x333)
        x333 = F.relu(x333)
        x333 = F.dropout(x333, p=self.dropout, training=self.training)



        x1 = self.layer31(x1, edge_index1)
        x2 = self.layer32(x2, edge_index2)
        x3 = self.layer33(x3, edge_index)
        x11 = self.layer31(x11, edge_index1)
        x22 = self.layer32(x22, edge_index2)
        x33 = self.layer33(x33, edge_index)
        x111 = self.layer31(x111, edge_index1)
        x222 = self.layer32(x222, edge_index2)
        x333 = self.layer33(x333, edge_index)


        x = x1 + x2 + x3 + x11 + x22 + x33 + x111 + x222 + x333
        # x = x1 + x2 + x3
        x = torch.sigmoid(x)

        return x
