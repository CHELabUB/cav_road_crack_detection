import argparse
import json
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import yaml
from datetime import datetime
from distutils.command.config import config
from config import get_config
from networks.LECSFormer import LECSFormer
from trainer import trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, 
                        default='../../../data/crack_detection/CrackTree260/',
                        help='data dir')
    parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
    parser.add_argument('--output_dirs', type=str, default='output/ablation/CrackTree260/', help='output dir')
    parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.001, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=list, default=[512,512], help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=44, help='random seed')
    parser.add_argument('--cfg', type=str, default='configs/config.yaml', metavar="FILE", help='path to config file', )
    parser.add_argument('--resume',help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help=" 'no: no cache,' 'full: cache all data,' 'part: sharding the dataset into nonoverlapping pieces and only cache one piece' ")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    args = parser.parse_args()
    return args


def get_augmentation_info(data_root):
    """Read augmentation info from preprocessing output"""
    aug_info_path = os.path.join(data_root, "augmentation_info.json")
    
    if os.path.exists(aug_info_path):
        with open(aug_info_path, 'r') as f:
            aug_info = json.load(f)
        return aug_info
    else:
        # Fallback to hardcoded if no metadata found
        return {
            "version": "unknown",
            "augmentation_flags": "unknown",
            "dataset_stats": "unknown"
        }

if __name__ == '__main__':
    args = parse_args()
    config = get_config(args)

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

    if not os.path.exists(args.output_dirs):
        os.makedirs(args.output_dirs)

    # Get augmentation info from preprocessing
    augmentation_info = get_augmentation_info(args.root_path)

    run_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.root_path,
        "augmentations": augmentation_info,
        "training_config": {
            "num_classes": args.num_classes,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.base_lr,
            "img_size": args.img_size,
            "seed": args.seed,
            "deterministic": bool(args.deterministic),
            "num_gpus": args.n_gpu,
        },
        "config_file": args.cfg
    }

    with open(os.path.join(args.output_dirs, "run_info.yaml"), "w") as f:
        yaml.dump(run_info, f, default_flow_style=False)

    print(f"Saved training configuration to {os.path.join(args.output_dirs, 'run_info.yaml')}")

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

    optimizer = optim.AdamW(model.parameters(), betas=config.TRAIN.OPTIMIZER.BETAS, eps=config.TRAIN.OPTIMIZER.EPS,
                            lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9, last_epoch=-1)
    
    start_epoch = 0
    best_f1 = 0.0
    best_threshold_overall = 0.5
    iter_num = 0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            
            # DEBUG: Print all keys in checkpoint
            print("ALL KEYS IN CHECKPOINT:")
            for key in checkpoint.keys():
                print(f"   '{key}': {type(checkpoint[key])}")
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model weights loaded successfully")
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded successfully")
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Scheduler state loaded successfully")
            
            # Load training state - explicitly check each key
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                scheduler.last_epoch = start_epoch  # Update scheduler's last epoch
                print(f"Resuming from epoch {start_epoch}")
            else:
                print("'epoch' not found in checkpoint, starting from epoch 0")
                
            if 'best_f1' in checkpoint:
                best_f1 = checkpoint['best_f1']
                print(f"Previous best F1: {best_f1:.4f}")
            else:
                print("'best_f1' not found in checkpoint, using 0.0")
                
            if 'best_threshold' in checkpoint:
                best_threshold_overall = checkpoint['best_threshold']
                print(f"Previous best threshold: {best_threshold_overall:.4f}")
            else:
                print("'best_threshold' not found in checkpoint, using 0.5")
                
            if 'iter_num' in checkpoint:
                iter_num = checkpoint['iter_num']
                print(f"Previous iteration: {iter_num}")
            else:
                print("'iter_num' not found in checkpoint, using 0")
                    
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Pass resume information to trainer
    resume_info = {
        'start_epoch': start_epoch,
        'best_f1': best_f1,
        'best_threshold': best_threshold_overall,
        'iter_num': iter_num,
        'optimizer': optimizer,
        'scheduler': scheduler
    }

    trainer(args, model, config, resume_info)
