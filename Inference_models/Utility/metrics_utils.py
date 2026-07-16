"""
Common metrics utilities for crack detection evaluation
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import logging
import os

def get_statistics_with_tolerance(pred, gt, tolerance=2):
    """
    Compute TP, FP, FN with tolerance.
    pred, gt must be binary {0,1}, single-channel 2D.
    """
    # Ensure binary 0/1 and 2D
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    if gt.ndim == 3 and gt.shape[0] == 1:
        gt = gt[0]  # squeeze channel dimension
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]

    # Now distanceTransform will work
    dist_gt = cv2.distanceTransform(1 - gt, cv2.DIST_L2, 3)
    dist_pred = cv2.distanceTransform(1 - pred, cv2.DIST_L2, 3)

    pred_tp = (pred == 1) & (dist_gt <= tolerance)
    gt_tp = (gt == 1) & (dist_pred <= tolerance)

    tp = np.sum(pred_tp)
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - np.sum(gt_tp)

    return tp, fp, fn

def calculate_metrics(pred_list, gt_list, thresholds, use_tolerance=True, tolerance=2):
    """Calculate precision, recall and F1 scores at different thresholds"""
    results = []
    per_image_all = {}

    # Flatten predictions and ground truths for evaluation
    all_preds = np.concatenate([p.flatten() for p in pred_list])
    all_gts = np.concatenate([g.flatten() for g in gt_list])
    
    # Calculate precision-recall curve
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(all_gts, all_preds)
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        f1_scores, precisions, recalls = [], [], []
        per_image_metrics = []

        for idx, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            binary_pred = (pred >= threshold).astype(np.uint8)
            if use_tolerance:
                tp, fp, fn = get_statistics_with_tolerance(binary_pred, gt, tolerance=tolerance)
            else:
                tp = np.sum((binary_pred == 1) & (gt == 1))
                fp = np.sum((binary_pred == 1) & (gt == 0))
                fn = np.sum((binary_pred == 0) & (gt == 1))

            img_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            img_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            img_f1        = 2 * img_precision * img_recall / (img_precision + img_recall) if (img_precision + img_recall) > 0 else 0

            f1_scores.append(img_f1)
            precisions.append(img_precision)
            recalls.append(img_recall)

            per_image_metrics.append({
                "image_id": idx,
                "precision": img_precision,
                "recall": img_recall,
                "f1": img_f1
            })

        results.append({
            'threshold': threshold,
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1_scores)
        })
        per_image_all[threshold] = per_image_metrics
    
    # ODS (best dataset-level F1)
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)
    ods = results_sorted[0]
    
    # OIS (average of best F1s per image)
    best_f1_per_image = []
    for pred, gt in zip(pred_list, gt_list):
        best_f1 = 0
        for threshold in thresholds:
            binary_pred = (pred >= threshold).astype(np.uint8)
            if use_tolerance:
                tp, fp, fn = get_statistics_with_tolerance(binary_pred, gt, tolerance=tolerance)
            else:
                tp = np.sum((binary_pred == 1) & (gt == 1))
                fp = np.sum((binary_pred == 1) & (gt == 0))
                fn = np.sum((binary_pred == 0) & (gt == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            best_f1 = max(best_f1, f1)
        best_f1_per_image.append(best_f1)
    ois = np.mean(best_f1_per_image)
    
    # Average Precision
    ap = average_precision_score(all_gts, all_preds)
    
    return results, ods, ois, ap, pr_precision, pr_recall, pr_thresholds, per_image_all

def plot_metrics(results, ods, ois, ap, output_path):
    """Plot metrics and save as image"""
    # Set global style
    plt.rcParams.update({
        "font.size": 16,          # base font size
        "axes.titlesize": 20,     # title size
        "axes.labelsize": 18,     # x/y label size
        "xtick.labelsize": 14,    # tick label size
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 3,     # thicker lines
        "lines.markersize": 8
    })

    plt.figure(figsize=(22, 10))  # make the figure slightly larger
    
    # --- Plot 1: Precision, Recall, F1 vs threshold ---
    plt.subplot(1, 2, 1)
    thresholds = [r['threshold'] for r in results]
    f1_scores = [r['f1'] for r in results]
    prec_scores = [r['precision'] for r in results]
    rec_scores = [r['recall'] for r in results]

    plt.plot(thresholds, f1_scores, 'b-', label="F1")
    plt.plot(thresholds, prec_scores, 'g--', label="Precision")
    plt.plot(thresholds, rec_scores, 'r-.', label="Recall")
    plt.scatter([ods['threshold']], [ods['f1']], color='black', s=120, zorder=5, label="ODS point")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, F1 vs Threshold')
    plt.legend()

    # --- Plot 2: Precision-Recall curve with AUC shading ---
    plt.subplot(1, 2, 2)

    # Use the custom calculated precision and recall values (with tolerance)
    custom_recalls = [r['recall'] for r in results]
    custom_precisions = [r['precision'] for r in results]

    # Sort by recall for proper PR curve plotting
    sorted_indices = np.argsort(custom_recalls)
    sorted_recalls = np.array(custom_recalls)[sorted_indices]
    sorted_precisions = np.array(custom_precisions)[sorted_indices]

    plt.plot(sorted_recalls, sorted_precisions, 'b-', label="PR Curve")
    plt.fill_between(sorted_recalls, sorted_precisions, alpha=0.2, color='blue')  # shaded AUC
    plt.scatter([ods['recall']], [ods['precision']], color='red', s=120, zorder=5, label="ODS")
    # Dynamically calculate annotation offset to keep it within plot bounds and avoid overlap
    offset_x = -0.15 if ods['recall'] > 0.8 else 0.05
    offset_y = -0.10 if ods['precision'] > 0.8 else 0.05
    ann_x = min(max(ods['recall'] + offset_x, 0.0), 1.0)
    ann_y = min(max(ods['precision'] + offset_y, 0.0), 1.05)
    plt.annotate(
        f"ODS: P={ods['precision']:.3f}, R={ods['recall']:.3f}",
        xy=(ods['recall'], ods['precision']),
        xytext=(ann_x, ann_y),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", linewidth=2),
        fontsize=14, backgroundcolor="white"
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP: {ap:.3f}, OIS: {ois:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # high-res export
    logging.info(f"Metrics plot saved to {output_path}")

def save_metrics(results, ods, ois, ap, per_image_all, case_names, metrics_save_path, precision=None, recall=None):
    """Save comprehensive metrics to files"""
    os.makedirs(metrics_save_path, exist_ok=True)
    
    # Save comprehensive metrics
    metrics_data = {
        'thresholds': [r['threshold'] for r in results],
        'f1_scores': [r['f1'] for r in results],
        'prec_scores': [r['precision'] for r in results],
        'rec_scores': [r['recall'] for r in results],
        'ods_threshold': ods['threshold'],
        'ods_f1': ods['f1'],
        'ods_precision': ods['precision'],
        'ods_recall': ods['recall'],
        'ois': ois,
        'ap': ap
    }
    
    if precision is not None and recall is not None:
        metrics_data['pr_precision'] = precision
        metrics_data['pr_recall'] = recall
    
    np.savez(os.path.join(metrics_save_path, 'comprehensive_metrics.npz'), **metrics_data)
    logging.info(f"Comprehensive metrics saved to {metrics_save_path}/comprehensive_metrics.npz")
    
    # Print results
    logging.info(f"ODS: F1={ods['f1']:.4f}, Precision={ods['precision']:.4f}, Recall={ods['recall']:.4f} at threshold {ods['threshold']:.2f}")
    logging.info(f"OIS: {ois:.4f}")
    logging.info(f"AP: {ap:.4f}")
    
    # Plot metrics
    plot_metrics(results, ods, ois, ap, os.path.join(metrics_save_path, 'metrics_plot.png'))
    
    # Save per-threshold metrics
    with open(os.path.join(metrics_save_path, 'metrics_results.csv'), 'w') as f:
        f.write("Threshold,Precision,Recall,F1\n")
        for r in results:
            f.write(f"{r['threshold']:.2f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")

    # Save per-image metrics at ODS threshold
    ods_thresh = ods['threshold']
    ods_per_image = per_image_all[ods_thresh]
    with open(os.path.join(metrics_save_path, 'per_image_metrics.csv'), 'w') as f:
        f.write("Image,Precision,Recall,F1\n")
        for idx, metrics in enumerate(ods_per_image):
            f.write(f"{case_names[idx]},{metrics['precision']:.4f},{metrics['recall']:.4f},{metrics['f1']:.4f}\n")
    logging.info("Per-image metrics saved.")

    # Identify best and worst images by F1
    sorted_images = sorted(zip(case_names, ods_per_image), key=lambda x: x[1]['f1'])
    worst_images = sorted_images[:3]
    best_images = sorted_images[-3:]

    logging.info(f"Worst 3 images: {[x[0] for x in worst_images]}")
    logging.info(f"Best 3 images: {[x[0] for x in best_images]}")

    # Save best threshold and image analysis
    with open(os.path.join(metrics_save_path, 'best_threshold.txt'), 'w') as f:
        f.write(f"Best threshold (ODS): {ods['threshold']:.4f}\n")
        f.write(f"ODS: F1={ods['f1']:.4f}, Precision={ods['precision']:.4f}, Recall={ods['recall']:.4f}\n")
        f.write(f"OIS: {ois:.4f}\n")
        f.write(f"AP: {ap:.4f}\n\n")

        f.write("Worst 3 images:\n")
        for name, m in worst_images:
            f.write(f"{name}: F1={m['f1']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}\n")

        f.write("\nBest 3 images:\n")
        for name, m in best_images:
            f.write(f"{name}: F1={m['f1']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}\n")

    return ods_per_image, worst_images, best_images
