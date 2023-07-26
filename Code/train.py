# 这个文件是专门用来训练预训练模型的

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_ACDC, trainer_synapse, trainer_la, trainer_Lits

parser = argparse.ArgumentParser()
#### 数据集切换
# parser.add_argument('--dataset', type=str,
#                     default='Synapse', help='experiment_name')
parser.add_argument('--dataset', type=str,
                    default='Lits', help='experiment_name')
#parser.add_argument('--dataset', type=str,
#                    default='LA', help='experiment_name')
### 初始化部分
parser.add_argument('--root_path', type=str,
                    default='/home/hehe/Medical_Program/TransUnet/project_TransUNet/data/ACDC',
                    help='root dir for data')
parser.add_argument('--list_dir', type=str,
                     default='./lists/lists_ACDC', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')

## 模型参数部分
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=1000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--label_batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input, LA set as 112')
parser.add_argument('--seed', type=int,
                    default=93, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--warmup_epoch', type=int, default=120, help="epoch num of warm up")
parser.add_argument('--ratio', type=float, default=0.2, help="label data ratio")
parser.add_argument('--fp16_precision', type=bool, default=True, help="fp16_precision")
parser.add_argument('--temperature', type=float, default=0.1, help="temperature for flc loss")
parser.add_argument('--T', type=int, default=8, help="T times for data argumentation")
parser.add_argument('--feature_channels', type=int, default=16, help="feature channels for feature uncertainty")
parser.add_argument('--pretrained_trunet', type=bool, default=True, help="feature channels for feature uncertainty")
parser.add_argument('--pretrained_epochs', type=int, default=120, help="feature channels for feature uncertainty")

args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'img_size': 224,
            'num_classes': 9,
        },
        'ACDC': {
            'root_path': '../data/ACDC',
            'list_dir': './lists/lists_acdc',
            'img_size': 224,
            'num_classes': 4,
        },
        'LA': {
            'root_path': "../data/LA",
            'list_dir': "./lists/list_LA",
            'img_size': 112,
            'num_classes': 2
        },
        'Lits': {
            'root_path': "../data/Lits",
            'list_dir': "./lists/list_lits",
            'img_size': 224,
            'num_classes': 3
        }
    }
    ####  下面两行代码原始代码里没有
    # if args.batch_size != 24 and args.batch_size % 6 == 0:
    #     args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.img_size = dataset_config[dataset_name]['img_size']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    # snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    # if args.pretrained_trunet is True:
    #     snapshot_path = snapshot_path + '_epo' + str(args.pretrained_epochs) if args.pretrained_epochs != 30 else snapshot_path
    # else:
    #     snapshot_path = snapshot_path + '_epo' + str(
    #         args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    ema_net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    if args.pretrained_trunet is True:
        pretrained_path = "../pretrained_model/{}/{}".format(args.exp, "TU_pretrain_R50-ViT-B_16_skip3_epo120_bs4_{}".format(args.img_size))#"TU_pretrain_R50-ViT-B_16_skip3_epo120_bs"+args.batch_size+"_224"
        pretrained_path = os.path.join(pretrained_path, 'best_model.pth')
        if not os.path.exists(pretrained_path): pretrained_path = pretrained_path.replace('best_model', 'epoch_' + str(119))
        net.load_state_dict(torch.load(pretrained_path))
        ema_net.load_state_dict(torch.load(pretrained_path))
        pretrained_path = pretrained_path.split('/')[-1]
        print("the pretrained model is : ", pretrained_path)

    trainer = {'ACDC': trainer_ACDC, 'Synapse': trainer_synapse, 'LA': trainer_la, 'Lits': trainer_Lits}
    trainer[dataset_name](args, net, ema_net, snapshot_path)
    # trainer[dataset_name](args, net, snapshot_path)
