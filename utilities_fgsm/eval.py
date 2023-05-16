#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

from typing import Iterable

import torch
from tqdm import tqdm

from utilities_fgsm.foward_pass import forward_pass, write_summary
from utilities_fgsm.misc import save_and_clear
from utilities_fgsm.summary_logger import TensorboardSummary
from utilities_fgsm.plot_frequencies import plot_frequencies

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, device: torch.device,
             epoch: int, summary: TensorboardSummary, save_output: bool, args=None, experiment_dir=None):
    model.eval()
    criterion.eval()
    feat_left, feat_right = [], []

    # initialize stats
    eval_stats = {'l1': 0.0, 'occ_be': 0.0, 'l1_raw': 0.0, 'iou': 0.0, 'rr': 0.0, 'epe': 0.0, 'error_px': 0.0,
                  'total_px': 0.0}
    # config text logger
    logger = summary.config_logger(epoch)
    # init output file
    if save_output:
        output_idx = 0
        output_file = {'left': [], 'right': [], 'disp': [], 'disp_pred': [], 'occ_mask': [], 'occ_pred': []}

    tbar = tqdm(data_loader)
    valid_samples = len(tbar)
    for idx, data in enumerate(tbar):
        # forward pass
        outputs, losses, sampled_disp, left_feats, right_feats = forward_pass(model, data, device, criterion, eval_stats, idx, logger, args)
        feat_left.append(left_feats.detach().cpu())
        feat_right.append(right_feats.detach().cpu())

        if losses is None:
            valid_samples -= 1
            continue

        # clear cache
        torch.cuda.empty_cache()

        # save output
        if save_output:
            output_file['left'].append(data['left'])
            output_file['right'].append(data['right'])
            output_file['disp'].append(data['disp'])
            output_file['occ_mask'].append(data['occ_mask'].cpu())
            output_file['disp_pred'].append(outputs['disp_pred'].data.cpu())
            output_file['occ_pred'].append(outputs['occ_pred'].data.cpu())

            # save to file
            if len(output_file['left']) >= 50:
                output_idx = save_and_clear(output_idx, output_file, experiment_dir)

    # save to file
    if save_output:
        save_and_clear(output_idx, output_file, experiment_dir)

    # compute avg
    eval_stats['epe'] = eval_stats['epe'] / valid_samples
    eval_stats['iou'] = eval_stats['iou'] / valid_samples
    eval_stats['px_error_rate'] = eval_stats['error_px'] / eval_stats['total_px']

    # write to tensorboard
    write_summary(eval_stats, summary, epoch, 'eval')

    # log to text
    logger.info('Epoch %d, epe %.4f, iou %.4f, px error %.4f' %
                (epoch, eval_stats['epe'], eval_stats['iou'], eval_stats['px_error_rate']))
    print()

    #import ipdb;ipdb.set_trace()
    logger.info('\n\n\n\n\nEXPERIMENT DIR={}\n\n\n\n'.format(args.experiment_dir))
    #plot_frequencies((feat_left, feat_right), args.experiment_dir, logger)


    return eval_stats
