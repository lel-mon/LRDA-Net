import argparse
import torch.nn as nn
import torch
import numpy as np


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_model():
    model_path = 'model.pth'
    model = torch.load(model_path)
    model.eval()
    return model


def evaluate_segmentation(predicted_mask, ground_truth):
    #  IoU, Dice coefficient
    intersection = np.logical_and(predicted_mask, ground_truth)
    union = np.logical_or(predicted_mask, ground_truth)
    iou = np.sum(intersection) / np.sum(union)

    dice = (2 * np.sum(intersection)) / (np.sum(predicted_mask) + np.sum(ground_truth))

    return {'iou': iou, 'dice': dice}
