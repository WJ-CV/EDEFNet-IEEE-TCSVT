import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=91, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=12, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch_list', type=int, default=[23, 53, 73], help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0, 1, 2, 3', help='train use gpu')

parser.add_argument('--start_weakly_epoch', type=float, default=36, help='Start using pseudo tags to refine')
parser.add_argument('--mask_num', type=float, default=10, help='mask_ratio = mask_num / 64')

parser.add_argument('--un_warm_epochs', type=float, default=15, help='Unsupervised weighted warm-up')
parser.add_argument('--un_warm_value', type=float, default=2, help='Unsupervised weights')

parser.add_argument('--weakly_warm_epochs', type=float, default=5, help='Weakly-supervised weighted warm-up')
parser.add_argument('--weakly_warm_value', type=float, default=2, help='Weakly-supervised weights')

# RGB-T Datasets   /data/wj/Train/supervised1
parser.add_argument('--sup_gt_root', type=str, default='/home/wj/RGBD dataset/Train/supervised/', help='the training GT images root')
parser.add_argument('--unsup_gt_root', type=str, default='/home/wj/RGBD dataset/Train/unsupervised/', help='the training GT images root')
parser.add_argument('--root_path', type=str, default='/home/wj/RGBD dataset/Train/', help='the training GT images root')

parser.add_argument('--test_gt_root', type=str, default='/home/wj/RGBD dataset/Val/', help='the training GT images root')
parser.add_argument('--test_root_path', type=str, default='/home/wj/RGBD dataset/Val/', help='the training GT images root')
parser.add_argument('--save_path', type=str, default='./model/', help='the path to save models')
opt = parser.parse_args()