import torch
import torch.nn as nn
from gaem import GAEM
import os


class Net(nn.Module):
    def __init__(self,args):
        super().__init__()
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            self.model = torch.load(args.weights)
            print('load dict:',args.weights)
        else:
            self.model = GAEM(
                chan = args.chan,
                attn_kernel_size = args.attn_kernel_size,
                dropout_rate= args.droprate,
                hidden=args.hid,
                num_layer=args.nlayer,
                )
            
        self.model.to(args.device[0])


    def forward(self,temp,lst = None):
        if lst == None:
            lst = temp
        occu = self.model(temp,lst)
        return occu

