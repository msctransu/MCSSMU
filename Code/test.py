import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_ACDC import ACDC_dataset
from datasets.dataset_la import LA_dataset
from datasets.dataset_lits import LITS_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()

#### 数据集选择
#parser.add_argument('--dataset', type=str,
#                    default='Synapse', help='experiment_name')
parser.add_argument('--dataset', type=str,
                      default='Lits', help='experiment_name')

#### 数据参数设置
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--num_classes', type=int,
                      default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                     default='/home/hehe/Medical_Program/TransUnet/project_TransUNet/data/ACDC/lists/lists_ACDC', help='list dir')

##### 模型参数设置
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True, action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=93, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'ACDC': {
            'Dataset': ACDC_dataset,
            'volume_path': '../data/ACDC/test',
            'list_dir': './lists/lists_acdc',
            'num_classes': 4,
            'z_spacing': 1,
            'img_size': 224
        },
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
            'img_size': 224
        },
        'LA': {
            'Dataset': LA_dataset,
            'volume_path': '../data/LA',
            'list_dir': './lists/list_LA',
            'num_classes': 2,
            'z_spacing': 1,
            'img_size': 112
        },
        'Lits': {
            'Dataset': LITS_dataset,
            'volume_path': '../data/Lits',
            'list_dir': './lists/list_lits',
            'num_classes': 3,
            'z_spacing': 1,
            'img_size': 224
        }
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.img_size = dataset_config[dataset_name]['img_size']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # print(net)
    # snapshot = os.path.join(snapshot_path, 'epoch_699.pth')  # mean_dice : 0.794571 mean_hd95 : 26.009214
    # snapshot = os.path.join(snapshot_path, 'epoch_749.pth')  # mean_dice : 0.776283 mean_hd95 : 20.367424
    # snapshot = os.path.join(snapshot_path, 'epoch_799.pth')  # mean_dice : 0.814956 mean_hd95 : 22.921424
    # snapshot = os.path.join(snapshot_path, 'epoch_849.pth')  # mean_dice : 0.812287 mean_hd95 : 25.992032
    # snapshot = os.path.join(snapshot_path, 'epoch_899.pth')  # mean_dice : 0.815832 mean_hd95 : 24.811601
    # snapshot = os.path.join(snapshot_path, 'epoch_949.pth')  # mean_dice : 0.817569 mean_hd95 : 25.713107
    # snapshot = os.path.join(snapshot_path, 'epoch_999.pth')  # mean_dice : 0.815204 mean_hd95 : 27.122285

    # snapshot = os.path.join(snapshot_path, 'epoch_999.pth')  # ACDC mean_dice : 0.912719 mean_hd95 : 1.691223
    # snapshot = os.path.join(snapshot_path, 'epoch_699.pth')  # ACDC mean_dice : 0.914735 mean_hd95 : 1.277291
    # snapshot = os.path.join(snapshot_path, 'epoch_549.pth')  # ACDC mean_dice : 0.914674 mean_hd95 : 1.157994
    # snapshot = os.path.join(snapshot_path, 'epoch_749.pth')  # ACDC mean_dice : 0.913248 mean_hd95 : 1.338973
    # snapshot = os.path.join(snapshot_path, 'epoch_799.pth')  # ACDC mean_dice : 0.913684 mean_hd95 : 1.306644
    # snapshot = os.path.join(snapshot_path, 'epoch_849.pth')  # ACDC mean_dice : 0.913075 mean_hd95 : 1.409595
    # snapshot = os.path.join(snapshot_path, 'epoch_949.pth')  # ACDC mean_dice : 0.912951 mean_hd95 : 1.489109
    #
    #snapshot = os.path.join(snapshot_path, 'best_model.pth')
    #if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    snapshot = os.path.join(snapshot_path, 'epoch_9549.pth')
    state_dict = torch.load(snapshot)
    from collections import OrderedDict
    new_state_item = OrderedDict()
    for k, v in state_dict.items():
        if k[:2] != "fc":
            new_state_item[k] = v
    # new_state_item["fc.0.weight"] = None
    #     print(k)
    # print("--------------------")
    # for k, v in net.named_parameters():
    #     print(k)

    net.load_state_dict(state_dict, strict=False)
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


