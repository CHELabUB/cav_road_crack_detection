# Changed trainer.py for plotting and training metrics calculation
import logging
import os
import sys
import time
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import summary
# from torch.nn.modules.loss import CrossEntropyLoss
from data.crack_datasets import Crack_Datasets
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc
from torch.utils.data import DataLoader
from tqdm import tqdm


def trainer(args, model, config, resume_info=None):
    
    logging.basicConfig(filename=args.output_dirs + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Initialize with resume info or defaults
    if resume_info is None:
        resume_info = {}
    
    start_epoch = resume_info.get('start_epoch', 0)
    best_f1 = resume_info.get('best_f1', 0.0)
    best_threshold_overall = resume_info.get('best_threshold', 0.5)
    iter_num = resume_info.get('iter_num', 0)
    optimizer = resume_info.get('optimizer', None)
    scheduler = resume_info.get('scheduler', None)
    
    # Log resume status
    if start_epoch > 0:
        logging.info(f"RESUMING TRAINING from epoch {start_epoch}")
        logging.info(f"Previous best F1: {best_f1:.4f}")
        logging.info(f"Previous best threshold: {best_threshold_overall:.4f}")
        logging.info(f"Starting from iteration: {iter_num}")
    else:
        logging.info("STARTING FRESH TRAINING")
    
    # Create optimizer and scheduler if not provided (fresh training)
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), betas=config.TRAIN.OPTIMIZER.BETAS, eps=config.TRAIN.OPTIMIZER.EPS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    
    if scheduler is None:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9, last_epoch=-1)

    # Setup metrics history for plotting
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'best_threshold': [],
        'pr_curves': []  # Store PR curve data for each evaluation epoch
    }

    train_data = Crack_Datasets(data_root=args.root_path,
                                img_list=os.path.join(args.root_path,'train.txt'),
                                img_size=args.img_size,
                                mode='train'
                                )
    val_data = Crack_Datasets(data_root=args.root_path,
                              img_list=os.path.join(args.root_path,'val.txt'),
                              img_size=args.img_size,
                              mode='val'
                              )
    train_loader = DataLoader(train_data,
                            batch_size=args.batch_size,
                            drop_last=True,
                            shuffle=True,
                            num_workers=12,
                            pin_memory=True)
    
    # Increased validation batch size
    val_batch_size = 4 
    val_loader = DataLoader(val_data,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=4)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    # Original BCE loss for comparison
    ce_loss = nn.BCEWithLogitsLoss()
    
    # New loss functions
    def dice_loss(pred, target, smooth=1.0):
        """
        Calculate Dice loss
        """
        pred = torch.sigmoid(pred)  # Apply sigmoid to convert logits to probabilities
        
        # Flatten the prediction and target tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice_score = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice_score

    def weighted_bce_loss(pred, target, weight_pos=5.0, weight_neg=1.0):
        """
        Weighted BCE that gives higher importance to positive pixels (cracks)
        """
        # Create weight tensor
        weights = torch.ones_like(target)
        weights[target == 1] = weight_pos  # Positive pixels get higher weight
        weights[target == 0] = weight_neg  # Negative pixels get lower weight
        
        # Apply weights to BCE loss
        bce = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
        weighted_bce = (weights * bce).mean()
        
        return weighted_bce

    def combined_loss(pred, target, w_bce=0.7, w_dice=0.3, pos_weight=5.0):
        """
        Combined weighted BCE and Dice loss
        """
        wbce = weighted_bce_loss(pred, target, weight_pos=pos_weight)
        dice = dice_loss(pred, target)
        return w_bce * wbce + w_dice * dice

    # Calculate max_iterations based on remaining epochs
    max_iterations = (args.max_epochs - start_epoch) * len(train_loader)
    model.train()
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    
    def calculate_metrics_with_threshold(outputs, labels, threshold=0.5):
        """Calculate precision, recall, and F1 score using a specific threshold"""
        # Convert logits to binary predictions
        predictions = (torch.sigmoid(outputs) > threshold).float()
        
        # Convert tensors to numpy arrays for sklearn
        predictions_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Calculate per image metrics and average
        batch_size = predictions_np.shape[0]
        precision_sum, recall_sum, f1_sum = 0, 0, 0
        valid_images = 0
        
        for i in range(batch_size):
            pred_flat = predictions_np[i].flatten()
            label_flat = labels_np[i].flatten()
            
            # Skip images with no positive pixels in ground truth
            if np.sum(label_flat) == 0:
                continue
                
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                label_flat, pred_flat, average='binary', zero_division=0
            )
            
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            valid_images += 1
        
        # Average metrics across valid images
        if valid_images == 0:
            return 0, 0, 0
            
        return (precision_sum/valid_images, recall_sum/valid_images, f1_sum/valid_images)
    
    def find_optimal_threshold(all_outputs, all_labels):
        """
        Find the threshold that gives the best F1 score
        Now uses an expanded range from 0.2 to 0.8
        """
        # Sample a subset of outputs and labels to reduce computation
        max_samples = min(len(all_outputs), 100)  # Limit to 100 samples max
        sample_indices = np.random.choice(len(all_outputs), size=max_samples, replace=False)
        
        sampled_outputs = [all_outputs[i] for i in sample_indices]
        sampled_labels = [all_labels[i] for i in sample_indices]
        
        # Use an expanded range of thresholds from 0.2 to 0.8 with more values
        thresholds = np.linspace(0.2, 0.8, 13)  # 13 evenly spaced values: 0.2, 0.25, 0.3, ... 0.8
        
        best_f1 = 0
        best_threshold = 0.5
        
        all_thresholds = []
        all_f1_scores = []
        all_precisions = []
        all_recalls = []
        
        for threshold in thresholds:
            threshold_precisions = []
            threshold_recalls = []
            threshold_f1s = []
            
            for output, label in zip(sampled_outputs, sampled_labels):
                precision, recall, f1 = calculate_metrics_with_threshold(output, label, threshold)
                threshold_precisions.append(precision)
                threshold_recalls.append(recall)
                threshold_f1s.append(f1)
            
            avg_precision = mean(threshold_precisions)
            avg_recall = mean(threshold_recalls)
            avg_f1 = mean(threshold_f1s)
            
            all_thresholds.append(threshold)
            all_f1_scores.append(avg_f1)
            all_precisions.append(avg_precision)
            all_recalls.append(avg_recall)
            
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = threshold
        
        # Create precision-recall-threshold plot
        pr_curve_data = {
            'thresholds': all_thresholds,
            'precisions': all_precisions,
            'recalls': all_recalls,
            'f1_scores': all_f1_scores,
            'best_threshold': best_threshold
        }
        
        return best_threshold, best_f1, pr_curve_data
    
    def create_full_pr_curve(all_outputs, all_labels):
        """
        Create full precision-recall curve across all probability thresholds
        """
        # Flatten all predictions and ground truths to create one large array
        all_preds = []
        all_targets = []
        
        for output, label in zip(all_outputs, all_labels):
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            targets = label.cpu().numpy().flatten()
            
            all_preds.append(probs)
            all_targets.append(targets)
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(all_targets, all_preds)
        
        # Calculate AUC-PR (Area Under Precision-Recall Curve)
        auc_pr = auc(recall, precision)
        
        # Return data for plotting
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'auc_pr': auc_pr
        }
    
    def plot_metrics(metrics_history, save_path):
        """Plot training progress and save to file - now with PR curves"""
        plt.figure(figsize=(20, 15))
        
        # Plot losses
        plt.subplot(3, 2, 1)
        plt.plot(metrics_history['train_loss'], label='Training Loss')
        plt.plot(metrics_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        
        # Plot precision and recall
        plt.subplot(3, 2, 2)
        plt.plot(metrics_history['val_precision'], label='Precision')
        plt.plot(metrics_history['val_recall'], label='Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title('Precision and Recall')
        plt.legend()
        plt.grid(True)
        
        # Plot F1 score
        plt.subplot(3, 2, 3)
        plt.plot(metrics_history['val_f1'], label='F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('F1 Score')
        plt.legend()
        plt.grid(True)
        
        # Plot optimal threshold
        plt.subplot(3, 2, 4)
        plt.plot(metrics_history['best_threshold'], label='Best Threshold')
        plt.xlabel('Epochs')
        plt.ylabel('Threshold')
        plt.title('Optimal Threshold Values')
        plt.legend()
        plt.grid(True)
        
        # Plot latest PR curve if available
        if metrics_history['pr_curves']:
            latest_pr_curve = metrics_history['pr_curves'][-1]
            
            # Plot PR threshold curve
            plt.subplot(3, 2, 5)
            plt.plot(latest_pr_curve['thresholds'], latest_pr_curve['precisions'], 'b-', label='Precision')
            plt.plot(latest_pr_curve['thresholds'], latest_pr_curve['recalls'], 'g-', label='Recall')
            plt.plot(latest_pr_curve['thresholds'], latest_pr_curve['f1_scores'], 'r-', label='F1 Score')
            plt.axvline(x=latest_pr_curve['best_threshold'], color='k', linestyle='--', 
                       label=f'Best Threshold: {latest_pr_curve["best_threshold"]:.2f}')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision, Recall, F1 vs Threshold')
            plt.legend()
            plt.grid(True)
            
            # Plot full PR curve if available
            if 'full_pr_curve' in latest_pr_curve:
                full_pr = latest_pr_curve['full_pr_curve']
                plt.subplot(3, 2, 6)
                plt.plot(full_pr['recall'], full_pr['precision'], 'b-')
                plt.fill_between(full_pr['recall'], full_pr['precision'], alpha=0.2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve (AUC = {full_pr["auc_pr"]:.3f})')
                plt.grid(True)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    # Training loop - start from start_epoch, not 0
    for epoch_num in tqdm(range(start_epoch, args.max_epochs), ncols=70):
        model.train()
        epoch_losses = []
        
        for i_batch, sample_batch in enumerate(train_loader):
            images, labels = sample_batch['image'], sample_batch['label']
            images, labels = images.cuda(), labels.cuda()

            output, mid_features = model(images)
            
            # Calculate combined loss for main output
            output_loss = combined_loss(output, labels, w_bce=0.7, w_dice=0.3, pos_weight=5.0)
            
            # Calculate loss for mid-features
            midfeatures_losses = []
            for i in range(len(mid_features)):
                mid_loss = combined_loss(mid_features[i], labels, w_bce=0.7, w_dice=0.3, pos_weight=5.0)
                midfeatures_losses.append(mid_loss)
            
            midfeatures_loss = sum(midfeatures_losses)
            
            # Combined loss
            loss = output_loss + midfeatures_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            lr_ = config.TRAIN.BASE_LR * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
                
            iter_num += 1
            epoch_losses.append(loss.item())
            
            if iter_num % 10 == 0:
                logging.info('iteration %d : loss : %f, mid_loss: %f, output_loss: %f' % (iter_num,
                                                                                    loss.item(),
                                                                                    midfeatures_loss.item(),
                                                                                    output_loss.item()))
        
        # Store average training loss for this epoch
        metrics_history['train_loss'].append(mean(epoch_losses))
        
        # Update scheduler at the end of each epoch
        scheduler.step()
        
        # Run validation only every 5 epochs (and on the final epoch)
        if (epoch_num + 1) % 5 == 0 or epoch_num == args.max_epochs - 1:
            logging.info(f"Running validation for epoch {epoch_num}")
            start_time = time.time()
            
            # Validation after selected epochs
            model.eval()
            val_losses = []
            all_outputs = []
            all_labels = []
            
            with torch.no_grad():
                for i_batch, sample_batch in tqdm(enumerate(val_loader), 
                                                  desc=f"Validation Epoch {epoch_num}",
                                                  total=len(val_loader)):
                    # Log progress regularly
                    if i_batch % 100 == 0:
                        logging.info(f"Validation batch {i_batch}/{len(val_loader)}")
                    
                    image, label = sample_batch['image'], sample_batch['label']
                    image, label = image.cuda(), label.cuda()
                    
                    output, mid_features = model(image)
                    # Use combined loss for validation too
                    loss_ = combined_loss(output, label)
                    val_losses.append(loss_.item())
                    
                    # Store all outputs and labels for threshold optimization
                    all_outputs.append(output)
                    all_labels.append(label)
            
            logging.info(f"Starting threshold optimization for epoch {epoch_num}")
            threshold_time_start = time.time()
            
            # Find optimal threshold from this epoch's validation data - with expanded range
            best_threshold, best_f1_value, pr_curve_data = find_optimal_threshold(all_outputs, all_labels)
            
            # Generate full PR curve
            full_pr_curve_data = create_full_pr_curve(all_outputs, all_labels)
            pr_curve_data['full_pr_curve'] = full_pr_curve_data
            
            # Store PR curve data
            metrics_history['pr_curves'].append(pr_curve_data)
            
            logging.info(f"Threshold optimization took {time.time() - threshold_time_start:.2f} seconds")
            logging.info(f"PR AUC: {full_pr_curve_data['auc_pr']:.4f}")
            
            # Use the best threshold to calculate final metrics
            logging.info(f"Calculating final metrics with threshold {best_threshold}")
            all_precisions = []
            all_recalls = []
            all_f1s = []
            
            for output, label in zip(all_outputs, all_labels):
                precision, recall, f1 = calculate_metrics_with_threshold(output, label, best_threshold)
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)
            
            avg_val_loss = mean(val_losses)
            avg_precision = mean(all_precisions)
            avg_recall = mean(all_recalls)
            avg_f1 = mean(all_f1s)
            
            # Fill in missing epochs in metrics history (for consistent plotting)
            # This is important for the plot x-axis alignment
            epochs_since_last_val = 1 if epoch_num == start_epoch else 5
            for _ in range(epochs_since_last_val):
                metrics_history['val_loss'].append(avg_val_loss)
                metrics_history['val_precision'].append(avg_precision)
                metrics_history['val_recall'].append(avg_recall)
                metrics_history['val_f1'].append(avg_f1)
                metrics_history['best_threshold'].append(best_threshold)
            
            # Log metrics
            logging.info(f"Epoch {epoch_num} Validation: Loss={avg_val_loss:.4f}, "
                         f"Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}, "
                         f"Best Threshold={best_threshold:.4f}")
            
            # Only create plots at major intervals
            if (epoch_num + 1) % 10 == 0 or epoch_num == args.max_epochs - 1:
                plot_metrics(metrics_history, os.path.join(args.output_dirs, f'metrics_epoch_{epoch_num}.png'))
            
            # Update the best overall threshold
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold_overall = best_threshold
                save_model_path = os.path.join(args.output_dirs, 'best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_threshold': best_threshold_overall,
                    'best_f1': best_f1,
                    'epoch': epoch_num,
                    'iter_num': iter_num
                }, save_model_path)
                logging.info(f"New best model saved with F1: {best_f1:.4f} at threshold: {best_threshold_overall:.4f}")
            
            logging.info(f"Validation for epoch {epoch_num} completed in {time.time() - start_time:.2f} seconds")

        # Regular saving schedule - Update to save optimizer and scheduler
        if (epoch_num + 1) % 10 == 0:
            save_model_path = os.path.join(args.output_dirs, f'epoch_{epoch_num}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch_num,
                'iter_num': iter_num,
                'best_f1': best_f1,
                'best_threshold': best_threshold_overall
            }, save_model_path)
            logging.info("save model to {}".format(save_model_path))

    # Final epoch saving - Save optimizer and scheduler
    save_model_path = os.path.join(args.output_dirs, f'final_epoch_{args.max_epochs-1}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_threshold': best_threshold_overall,
        'epoch': args.max_epochs-1,
        'iter_num': iter_num
    }, save_model_path)
    logging.info("save model to {}".format(save_model_path))
    
    # Save final metrics plot
    plot_metrics(metrics_history, os.path.join(args.output_dirs, 'final_metrics.png'))
    
    # Save metrics history as CSV for further analysis
    metrics_df = pd.DataFrame({
        'train_loss': metrics_history['train_loss'],
        'val_loss': metrics_history['val_loss'],
        'val_precision': metrics_history['val_precision'],
        'val_recall': metrics_history['val_recall'],
        'val_f1': metrics_history['val_f1'],
        'best_threshold': metrics_history['best_threshold']
    })
    metrics_df.to_csv(os.path.join(args.output_dirs, 'metrics_history.csv'), index_label='epoch')
    
    # Save PR curve data separately
    with open(os.path.join(args.output_dirs, 'pr_curves.pkl'), 'wb') as f:
        pickle.dump(metrics_history['pr_curves'], f)
    
    logging.info(f"Training completed. Best model had F1={best_f1:.4f} at threshold={best_threshold_overall:.4f}")
    
    return model, best_threshold_overall
