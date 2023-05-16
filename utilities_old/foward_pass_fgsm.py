#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch

from utilities.misc import NestedTensor


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    #perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def forward_pass(model, data, device, criterion, stats, idx=0, logger=None, epsilon:float=0.0):
    """
    forward pass of the model given input
    """
    # read data
    left, right = data['left'].to(device), data['right'].to(device)
    disp, occ_mask, occ_mask_right = data['disp'].to(device), data['occ_mask'].to(device), \
                                     data['occ_mask_right'].to(device)

    bs, _, h, w = left.size()

    # build the input
    inputs = NestedTensor(left, right, disp=disp, occ_mask=occ_mask, occ_mask_right=occ_mask_right)

    inputs.left.requires_grad=True
    inputs.right.requires_grad=True
    #inputs.occ_mask.requires_grad=True
    #inputs.occ_mask_right.requires_grad=True
    inputs.disp.requires_grad=True

    # forward pass
    with torch.enable_grad():
        outputs, feat_left, feat_right = model(inputs)

        # compute loss
        losses = criterion(inputs, outputs)

    #import ipdb;ipdb.set_trace()   


    if losses is None:
        return outputs, losses, disp
    
    
    losses['aggregated'].retain_grad()
    #losses['aggregated'].requires_grad=True
    losses['aggregated'].backward()
    
    
    #logger.info("ATTACKING WITH FGSM")
    inputs.left = fgsm_attack(inputs.left, epsilon, inputs.left.grad)
    right.left = fgsm_attack(inputs.right, epsilon, inputs.right.grad)

    with torch.no_grad():
        outputs, feat_left, feat_right = model(inputs)
    
    # compute loss
    losses = criterion(inputs, outputs)
    if losses is None:
        return outputs, losses, disp


    # get the loss
    stats['rr'] += losses['rr'].item()
    stats['l1_raw'] += losses['l1_raw'].item()
    stats['l1'] += losses['l1'].item()
    stats['occ_be'] += losses['occ_be'].item()

    stats['iou'] += losses['iou'].item()
    stats['epe'] += losses['epe'].item()
    stats['error_px'] += losses['error_px']
    stats['total_px'] += losses['total_px']

    # log for eval only
    if logger is not None:
        logger.info('Index %d, l1_raw %.4f, rr %.4f, l1 %.4f, occ_be %.4f, epe %.4f, iou %.4f, px error %.4f' %
                    (idx, losses['l1_raw'].item(), losses['rr'].item(), losses['l1'].item(), losses['occ_be'].item(),
                     losses['epe'].item(), losses['iou'].item(), losses['error_px'] / losses['total_px']))

    return outputs, losses, disp, (feat_left, feat_right, left, right)
