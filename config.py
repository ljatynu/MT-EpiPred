import argparse

parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)') 
parser.add_argument('--batch_size_eval', type=int, default=128,
                    help='input batch size for test (default: 128)')                     

parser.add_argument('--lr', type=float, default=0.00005,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0.002,
                    help='weight decay (default: 0.002)')

parser.add_argument('--emb_dim', type=int, default=1024,
                    help='embedding dimensions (default: 300)')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio (default: 0.5)')

parser.add_argument('--dataset', type=str, default = 'HME_datatset', help='root directory of dataset. For now, only classification.')
parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')


parser.add_argument('--layer_size', type=str, default='moderate',
                    help='size of bond(shared) layers(shallow, moderate, deep, task_relate), default "deep"')

args = parser.parse_args()

 # there are 78 protein in HDM dataset
args.train_percent = 80
args.path_head = 'dataset/HME_dataset/split/'
args.epoch = 1000
args.task_no = range(78)
args.num_tasks = len(args.task_no)
