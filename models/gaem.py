import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.modules.transformer import _get_clones

from models.attention import Attention
from models.aggregation import GCN

'''
data: batch, context size, ancients + current node (4), level + octant + occ (3)
'''

class ParentGraphGeneration(nn.Module):
    def __init__(self):
        super(ParentGraphGeneration, self).__init__()

    def forward(self, data):

        matrix = torch.bmm(data,data.transpose(-2,-1))
        matrix = torch.sqrt(matrix)
        matrix = matrix-torch.trunc(matrix)
        re1 = torch.tensor(0).to(data.device)
        re2 = torch.tensor(1).to(data.device)
        matrix = torch.where(matrix==0,re2,re1).to(torch.float)
        return matrix


class DistanceGraphGeneration(nn.Module):
    def __init__(self):
        super(DistanceGraphGeneration, self).__init__()
        
    def forward(self, src):

        theta = src[:,:,0].unsqueeze(-1)
        phi = src[:,:,1].unsqueeze(-1)
        radius = src[:,:,2].unsqueeze(-1)

        h = radius* torch.sin(phi)
        x_2plusy_2 = h^2 + (h^2).transpose(-1,-2) - h*(h.transpose(-1,-2)) * torch.cos(theta - theta.transpose(-1,-2))
        z_2 = (h * torch.tan(theta) - (h * torch.tan(theta)).transpose(-1,-2))^2

        dis = x_2plusy_2 + z_2

        dis = (dis - dis.min())/(dis.max() - dis.min())
        dis = 1-dis
        return dis


class GAttnEncoderLayer(Module):

    def __init__(
            self, 
            d_model=256,
            attn_kernel_size=16, 
            dropout = 0.1, 
            layer_norm_eps = 1e-5,
            ):

        super(GAttnEncoderLayer, self).__init__()

        self.attn = Attention(d_model, attn_kernel_size)

        self.dropout1 = nn.Dropout(dropout)

        self.dropout2 = nn.Dropout(dropout)


        self.aft_attn_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),   
            )
        
        self.norm1 = nn.LayerNorm(d_model,layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model,layer_norm_eps)

    def _ga_block(self, x, x_lst, adj, dis):

        x = x.transpose(-2,-1)

        x, x_lst = self.attn(x, x_lst, adj, dis)

        x = x.transpose(-2,-1)

        return self.dropout1(x), self.dropout2(x_lst)

    # only update the x and x_lst in Graph Attention Block
    def forward(self, src, x_lst, adj, dis):

        x = src

        x_1, x_lst = self._ga_block(x, x_lst, adj, dis)

        x = self.norm1(x + x_1)

        x = self.norm2(x + self.aft_attn_mlp(x))
        
        return x, x_lst
    

class GAttnEncoder(Module):

    def __init__(
            self,
            encoder_layer, 
            num_layers, 
            ):
        
        super(GAttnEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers,)
        
    def forward(self, src, lst, par, dis):

        output = src

        # only update the output and lst in Graph Attention Block
        for intr, mod in enumerate(self.layers):

            output, lst = mod(output, lst, par, dis)

        return output

class Embed_Block(Module):

    def __init__(
            self,
            input_dim, 
            d_model=128,
            ):
        
        super(Embed_Block, self).__init__()

        self.get_adj = ParentGraphGeneration()
        self.get_dis = DistanceGraphGeneration()
        
        self.embedding = GCN(input_dim,d_model)
        
    def forward(self, src):

        adj1 = src[:,:,3].unsqueeze(-1) # first parent index

        adj2 = src[:,:,4].unsqueeze(-1) # second parent index
        
        distance_graph = self.get_dis(src)

        par_graph = (self.get_adj(adj1) + self.get_adj(adj2))/2

        output = self.embedding(src, par_graph, distance_graph)

        return output, par_graph, distance_graph

class GAEM(nn.Module):

    def __init__(
            self, 
            chan,
            attn_kernel_size, 
            dropout_rate, 
            hidden=256, 
            num_layer=3,
            ):
        
        print(
            "GAEM: chan: {},  num_layer: {}, hidden: {}, dropout: {}".format(
                chan,
                num_layer,
                hidden,
                dropout_rate,
            ))
        
        super(GAEM, self).__init__()
        
        self.gcn = GCN(input_dim=hidden,out_features=hidden)

        self.Embed_Block = Embed_Block(
            input_dim=chan, 
            d_model=hidden,
            )

        self.encoder_layer = GAttnEncoderLayer(
            d_model = hidden,
            attn_kernel_size = attn_kernel_size, 
            dropout = dropout_rate, 
            layer_norm_eps = 1e-5,
            )

        self.Encode_block = GAttnEncoder(
            self.encoder_layer, 
            num_layers=num_layer, 
            )
        
        self.lst_MLP = nn.Sequential(
            nn.Linear(chan+1, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
        )
        
        self.output_MLP = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),

        )

    def forward(self, features,lst):

        lst = self.lst_MLP(lst)

        features, par, dis= self.Embed_Block(features)
        
        features = self.Encode_block(features, lst, par, dis)
        
        out = self.output_MLP(features)
        return out


if __name__ == '__main__':
    model = GAEM(
        chan = 8,
        attn_kernel_size = 32, 
        dropout_rate = 0.5, 
        hidden = 256,
        )
    currrent = torch.zeros((64, 16, 8))  # batch_size =  64, sequence = 16, dimention=8
    print(model(currrent))
    # torch.save(model,"1.pth")
