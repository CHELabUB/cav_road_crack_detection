import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
import logging
import sys
sys.path.append('../')
from Utility.metrics_utils import calculate_metrics, save_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def test(test_data_path, save_path, pretrained_model, tolerance):
    
    # Setup logging
    log_folder = './test_log/deepcrack_test_log'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_folder, 'deepcrack_test.txt'), 
        level=logging.INFO, 
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    # Log arguments
    logging.info(f"Test data path: {test_data_path}")
    logging.info(f"Save path: {save_path}")
    logging.info(f"Pretrained model: {pretrained_model}")
    logging.info(f"Tolerance: {tolerance}")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_pipline = dataReadPip(transforms=None)
    test_list = readIndex(test_data_path)
    test_dataset = loadedDataset(test_list, preprocess=test_pipline)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=1, drop_last=False)

    # -------------------- build trainer --------------------- #
    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)
    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=True))
    model.eval()

    # Instead of flattening labels and preds
    all_preds = []
    all_labels = []
    case_names = []

    with torch.no_grad():
        for names, (img, lab) in tqdm(zip(test_list, test_loader)):
            test_data, test_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
            test_pred = trainer.val_op(test_data, test_target)
            test_pred = torch.sigmoid(test_pred[0].cpu().squeeze())
            gt_mask = lab.cpu().squeeze()

            # Save images
            save_pred = torch.zeros((512 * 2, 512))
            save_pred[:512, :] = test_pred
            save_pred[512:, :] = gt_mask
            save_name = os.path.join(save_path, os.path.split(names[1])[1])
            save_pred = save_pred.numpy() * 255
            cv2.imwrite(save_name, save_pred.astype(np.uint8))

            # Append raw predictions and GTs for pixel-level metric analysis
            all_preds.append(test_pred.numpy())
            all_labels.append(gt_mask.numpy())
            case_names.append(os.path.split(names[1])[1].split('.')[0])

    # Create metrics directory
    metrics_save_path = os.path.join(save_path, "metrics")
    
    # Calculate metrics using common utilities
    thresholds = np.linspace(0, 1, 101)[1:-1]

    results, ods, ois, ap, precision, recall, _, per_image_all = calculate_metrics(
        all_preds, all_labels, thresholds, use_tolerance=(tolerance > 0), tolerance=tolerance)
    
    # Use common save_metrics function
    save_metrics(results, ods, ois, ap, per_image_all, case_names, metrics_save_path, precision, recall)
    
    return "Testing Finished!"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepCrack Testing Script')
    
    # Required arguments
    parser.add_argument('--test_data_path', type=str, default='data/datasets/CRKWH100/test.txt',
                       help='Path to test data index file')
    
    parser.add_argument('--pretrained_model', type=str, default='checkpoints/DeepCrack_CT260_FT1.pth',
                       help='Path to pretrained model file')
    
    parser.add_argument('--save_path', type=str, default='deepcrack_results/test_run/',
                       help='Directory to save results')
    
    parser.add_argument('--tolerance', type=int, default=2,
                       help='Pixel tolerance for F1 computation')
    
    args = parser.parse_args()
    
    test(
        test_data_path=args.test_data_path,
        save_path=args.save_path,
        pretrained_model=args.pretrained_model,
        tolerance=args.tolerance
    )
