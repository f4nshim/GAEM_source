import warnings
import torch.nn as nn
import torch
import torch.nn.functional as F

# Cross-Attention
class Cross_Attention(nn.Module):

        def __init__(self, channel=256):
            super().__init__()
            self.q=nn.Conv1d(channel,channel//2,kernel_size=(1))
            self.k=nn.Conv1d(channel,1,kernel_size=(1))
            self.softmax=nn.Softmax(1)
            self.z=nn.Conv1d(channel//2,channel,kernel_size=(1))
            self.ln=nn.LayerNorm(channel)
            self.sigmoid=nn.Sigmoid()


        def forward(self, x, y):
            b, c, _ = x.size()
            # the cross-attention branch
            q = self.q(x) #bs,c//2,w
            k = self.k(y) #bs,1,w
            k = k.reshape(b,-1,1) #bs,w,1
            k = self.softmax(k)
            AttnMatrix = torch.matmul(q,k) #bs,c//2,1
            AttnW = self.sigmoid(self.ln(self.z(AttnMatrix).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1) #bs,c,1,1
            out_x = AttnW * x
            out_y = AttnW * y
            return out_x, out_y

# Group-graph Self-Attention
class Graph_branch(nn.Module):
    def __init__(self, in_features, out_features, attn_kernel_size, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.attn_kernel_size = attn_kernel_size
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.norm = nn.LayerNorm(out_features)
        self.aftermlp = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features),   
            )
        
    def forward(self, h, adj):

        Wh = torch.matmul(h, self.W) # h.shape: (B, window size N, in_features), Wh.shape: (B,N, out_features)

        b,n,f = Wh.shape

        a_input = self._prepare_attentional_mechanism_input(Wh, adj, self.attn_kernel_size) # (B, H, N//H, N//H, 2*out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        if len(adj.shape)==3:
            adj = self._prepare_multihead_adj(adj,self.attn_kernel_size)

        adj = adj * e

        adj = F.softmax(adj, dim=-1)

        # sum of attention scalar
        Wh = Wh.reshape(b,n//self.attn_kernel_size,self.attn_kernel_size,f)

        output = torch.matmul(adj, Wh)
        output = self.norm(self.aftermlp(output) + output)
        output = output.reshape(b,n,f)
        
        return output
        
    def _prepare_attentional_mechanism_input(self, Wh, adj, attn_kernel_size, therehold = 0.2):
        B,N,F = Wh.size()

        # trick: the therehold split to examine attn_kernel_size
        x_axis = range(0,N-attn_kernel_size,attn_kernel_size)
        y_axis = range(attn_kernel_size-1,N-1,attn_kernel_size)
        mean = torch.mean(adj[x_axis,y_axis])
        if mean<therehold:
            warnings.warn("attn kernel size too large", SyntaxWarning)
        
        # split the context to sub-windows
        Wh = Wh.reshape(B,N//attn_kernel_size,attn_kernel_size,F)

        Wh_repeated_in_chunks = torch.repeat_interleave(Wh,attn_kernel_size, dim=2) #(B,H, N*N//H//H, out_features)

        # all_combinations_matrix.shape == (B, H, N*N//H//H, out_features)

        return Wh_repeated_in_chunks.view(B, N//attn_kernel_size, attn_kernel_size, attn_kernel_size, self.out_features)
    
    def _prepare_multihead_adj(self, adj, attn_kernel_size):

        B,N,_ = adj.shape

        out_adj = []

        for i in range(N//attn_kernel_size):

            out_adj.append(adj[:, i*attn_kernel_size:(i+1)*attn_kernel_size , i*attn_kernel_size:(i+1)*attn_kernel_size].unsqueeze(1))

        out_adj = torch.cat(out_adj, dim = 1)
        return out_adj


class Attention(nn.Module):

    def __init__(self, channel=256, attn_kernel_size=16):
         super().__init__()
         self.c = Cross_Attention(channel)
         self.g = Graph_branch(channel, channel, attn_kernel_size)
         self.norm = nn.LayerNorm(channel)
         

    def forward(self,x, y, adj1, adj2):
         
         x1, y = self.c(x,y)

         x2 = self.norm(self.g(x, adj1) + self.g(x,adj2) + x)

         x = x1 + x2

         return x, y

