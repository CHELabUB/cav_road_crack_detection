# Default Python Packages
import argparse
import logging
import os
import random
import sys
# Installed Packages
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# Packages in this repo
from config import get_config
from data.crack_datasets import Crack_Datasets
from networks.LECSFormer import LECSFormer
sys.path.append('../')
from Utility.metrics_utils import calculate_metrics, save_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../../data/crack_detection/Real-time_data_111/',
                    help='data dir')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='output/crack_seg_11300/Real-time_111/', help='output dir')
parser.add_argument('--checkpoints', type=str, default='output/crack_seg_11300/best_model.pth', help='weights')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--img_size', type=list, default=[512, 512], help='input patch size of network input')
parser.add_argument('--seed', type=int, default=44, help='random seed')
parser.add_argument('--cfg', type=str, default='configs/config.yaml', metavar="FILE", help='path to config file', )
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, ' 'full: cache all data,' 'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--tolerance', type=int, default=2, help='pixel tolerance for F1 computation')
args = parser.parse_args()
config = get_config(args)

def inference(args, model, test_save_path=None):
    test_data = Crack_Datasets(data_root=args.root_path,
                               img_list=os.path.join(args.root_path, 'test.txt'),
                               img_size=args.img_size,
                               mode='test')
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    
    all_predictions, all_ground_truths, case_names = [], [], []
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['name'][0]
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad():
            outputs, mid_fea = model(image)
            out = torch.sigmoid(outputs).squeeze(0)
            out_np = out.detach().cpu().numpy()

            all_predictions.append(out_np)
            gt_np = label.squeeze(0).cpu().numpy()
            gt_np = (gt_np > 0).astype(np.uint8)   # ensure binary 0/1, uint8
            all_ground_truths.append(gt_np)
            case_names.append(case_name)
            
            if len(out_np.shape) > 2:
                if out_np.shape[0] == 1:
                    out_np = out_np[0]
                else:
                    out_np = out_np[0]
            out_img = (out_np * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(test_save_path, case_name + '.png'), out_img)
    
    metrics_save_path = os.path.join(args.output_dir, "metrics")
    thresholds = np.linspace(0, 1, 101)[1:-1]

    results, ods, ois, ap, precision, recall, _, per_image_all = calculate_metrics(
        all_predictions, all_ground_truths, thresholds, use_tolerance=(args.tolerance > 0), tolerance=args.tolerance)
    
    # Use common save_metrics function
    save_metrics(results, ods, ois, ap, per_image_all, case_names, metrics_save_path, precision, recall)
    
    return "Testing Finished!"

if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = LECSFormer(img_size=args.img_size,
                    patch_size=config.MODEL.LECSFormer.PATCH_SIZE,
                    in_channels=config.MODEL.LECSFormer.IN_CHANS,
                    num_classes=args.num_classes,
                    embed_dim=config.MODEL.LECSFormer.EMBED_DIM,
                    depths=config.MODEL.LECSFormer.DEPTHS,
                    num_heads=config.MODEL.LECSFormer.NUM_HEADS,
                    window_size=config.MODEL.LECSFormer.WINDOW_SIZE,
                    mlp_ratio=config.MODEL.LECSFormer.MLP_RATIO,
                    qkv_bias=config.MODEL.LECSFormer.QKV_BIAS,
                    qk_scale=config.MODEL.LECSFormer.QK_SCALE,
                    drop_rate=config.MODEL.DROP_RATE,
                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                    patch_norm=config.MODEL.LECSFormer.PATCH_NORM,
                    use_checkpoint=config.TRAIN.USE_CHECKPOINT
                    ).cuda()
    
    # Load checkpoint first to check its structure
    check_points = torch.load(args.checkpoints)
    
    # Determine if checkpoint was saved with DataParallel during training
    if isinstance(check_points, dict) and 'model_state_dict' in check_points:
        state_dict = check_points['model_state_dict']
        # Check if first key has 'module.' prefix
        first_key = next(iter(state_dict.keys()))
        use_data_parallel = first_key.startswith('module.')
    else:
        # Direct state dict
        first_key = next(iter(check_points.keys()))
        use_data_parallel = first_key.startswith('module.')
    
    print(f"Checkpoint uses DataParallel: {use_data_parallel}")
    
    # Wrap with DataParallel ONLY if checkpoint was saved with DataParallel
    if use_data_parallel:
        model = nn.DataParallel(model)
        print("Using DataParallel wrapper")
    else:
        print("Not using DataParallel wrapper")
    
    # Now load the weights
    if isinstance(check_points, dict) and 'model_state_dict' in check_points:
        msg = model.load_state_dict(check_points['model_state_dict'], strict=True)
        best_threshold = check_points.get('best_threshold', 0.5)
    else:
        msg = model.load_state_dict(check_points, strict=True)
    
    print(f"Model loaded successfully! Loading message: {msg}")
    
    checkpoints_name = args.checkpoints.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+ checkpoints_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(checkpoints_name)

    test_save_path = os.path.join(args.output_dir, "test")
    os.makedirs(test_save_path, exist_ok=True)
    inference(args, model, test_save_path)
