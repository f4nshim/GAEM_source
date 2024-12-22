import argparse   

####################################### train_parser for SSEM
train_parser = argparse.ArgumentParser()
# dataset settings
train_parser.add_argument('--train_dataset',type=list, default=[
    'octree/KITTI/15_1024/train/', # for SemanticKITTI 
    ],
                                                            help='datasets use fot train')
train_parser.add_argument('--train_range',type=list,default=[0,0.9],help='the percentages used for train datasets (i.e 0-1.0)')
train_parser.add_argument('--val_dataset', type=list, default=[
    'octree/KITTI/15_1024/train/', # for SemanticKITTI 
    ],
                                                            help='datasets use fot validate')


train_parser.add_argument('--val_range',type=list,default=[0.9,1],help='the percentages used for val datasets (i.e 0-1.0)')
train_parser.add_argument('--num_workers',type=int,default=0)

# model settings
train_parser.add_argument('--graph_embedding', type=bool, default=True)
train_parser.add_argument('--graph_encoder', type=bool, default=True)
train_parser.add_argument('--channel_conv', type=bool, default=True)

train_parser.add_argument('--save_weights', type=str, default="./weights/KITTI/", help='en&decoder weights place')
train_parser.add_argument('--chan', type = int, default=8, help='input features of model')
train_parser.add_argument('--hid', type=int, default = 288, help='embedded size')
train_parser.add_argument('--attn_kernel_size', type=int, default = 8, help='embedded size')
train_parser.add_argument('--nlayer', type = list, default=[1,4,1], help='number of layers of en&decoder')
train_parser.add_argument('--droprate', type = float, default=0.1, help='')
# train settings
train_parser.add_argument('--epochs', type=int, default=15)
train_parser.add_argument('--batch_size', type=int, default=64) 
train_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
train_parser.add_argument('--weights', type=str, default='./weights/KITTI/288_[1, 4, 1]_0.0001/model.pth', help='the saved model which can be used for continuing training')
train_parser.add_argument('--device',type=list, default=['cuda:0'], help='device id (i.e. 0 or 0,1 or cpu)')




########################### test_parser
test_parser = argparse.ArgumentParser()
test_parser.add_argument('--dataset_type',type=str,default='KITTI',help='your prepared dataset, KITTI or ScanNet, etc.')
test_parser.add_argument('--mode',type=str,default='test', help='mode to store res or directly use (i.e. train or test)')

# # dataset settings
test_parser.add_argument('--seqsize', type = int, default=1024, help='')
test_parser.add_argument('--chan', type = int, default=8, help='input features of model')
test_parser.add_argument('--hid', type=int, default = 256, help='embedded size')
test_parser.add_argument('--attn_kernel_size', type=int, default = 32, help='embedded size')
test_parser.add_argument('--nlayer', type = int, default=3, help='number of layers of en&decoder')
test_parser.add_argument('--droprate', type = float, default=0.1, help='')
test_parser.add_argument('--test_dataset', type=list, default=[
    # '../../octree/ScanNet/13_1024/test/', # for ScanNet dataset
    'octree/KITTI/13_1024/train/', # for SemanticKITTI 
    ],
                                                            help='datasets use fot validate')
test_parser.add_argument('--test_range',type=list,default=[[0,0.01]],help='the percentages used for val datasets (i.e 0-1.0)')
test_parser.add_argument('--num_workers',type=int,default=0)

# test settings
test_parser.add_argument('--batch_size', type=int, default=32)
test_parser.add_argument('--weights', type=str, default='./weights/KITTI/288_[1, 4, 1]_0.0001/model.pth', help='en&decoder weights place') # ./weights/ScanNet/model.pth
test_parser.add_argument('--device',type=list, default=['cuda:0'], help='device id (i.e. cuda: or cpu)')

# # dataset settings
generate_parser = argparse.ArgumentParser()
generate_parser.add_argument('--seqsize', type = int, default=1024, help='advised 64, 128, 256, 512, 1024, 2048, etc.')
generate_parser.add_argument('--depth', type=int, default = 14, help='if KITTI then advised 10-15, if ScanNet then advised 6-11')
