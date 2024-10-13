import torch.nn as nn
import torch

# Our purposed GraphConv
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.kaiming_uniform_(self.weight.data) 
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs, adj):
        x = inputs
        x = torch.matmul(x,self.weight)
        x = torch.matmul(adj, x)
        outputs = self.norm(x)
        return outputs

class GCN(nn.Module):
    def __init__(self, input_dim, out_features):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(input_dim, out_features) 
        self.conv2 = GraphConv(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)

        self.aft_GCN = nn.Sequential(
            nn.Linear(input_dim, out_features),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features),
        )

    
    def forward(self, X, adj1, adj2):
        # X = [t,p,r,faid1, faid2, depth,index,parent_occu]

        X = self.norm(self.conv1(X,adj1) + self.conv2(X,adj2))
        X = self.aft_GCN(X)

        return X


    






    