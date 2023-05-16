#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch
import torch.nn.functional as F

from utilities.misc import NestedTensor


def write_summary(stats, summary, epoch, mode):
    """
    write the current epoch result to tensorboard
    """
    summary.writer.add_scalar(mode + '/rr', stats['rr'], epoch)
    summary.writer.add_scalar(mode + '/l1', stats['l1'], epoch)
    summary.writer.add_scalar(mode + '/l1_raw', stats['l1_raw'], epoch)
    summary.writer.add_scalar(mode + '/occ_be', stats['occ_be'], epoch)
    summary.writer.add_scalar(mode + '/epe', stats['epe'], epoch)
    summary.writer.add_scalar(mode + '/iou', stats['iou'], epoch)
    summary.writer.add_scalar(mode + '/3px_error', stats['px_error_rate'], epoch)

# FGSM attack code
def fgsm_attack(image, epsilon, alpha, data_grad, clip_min, clip_max):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + alpha*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, clip_min, clip_max)
    # Return the perturbed image
    return perturbed_image


def forward_pass(model, data, device, criterion, stats, idx=0, logger=None, args=None):
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

    """halfing the torch models and tensors for ICML"""
    model.half()
    for p in model.parameters():
        p.half()
    inputs.left = inputs.left.half()
    inputs.right = inputs.right.half()
    inputs.disp = inputs.disp.half()

    left_clip_min, left_clip_max = inputs.left.min()-args.epsilon, inputs.left.max()+args.epsilon
    right_clip_min, right_clip_max = inputs.right.min()-args.epsilon, inputs.right.max()+args.epsilon

    if args.attack == 'cospgd' or args.attack == 'pgd':
        inputs.left = inputs.left + torch.FloatTensor(inputs.left.shape).uniform_(-1*args.epsilon, args.epsilon).cuda()
        inputs.right = inputs.right + torch.FloatTensor(inputs.right.shape).uniform_(-1*args.epsilon, args.epsilon).cuda()
        if args.iterations ==1:
            alpha = args.epsilon
        else:
            alpha = args.alpha
        inputs.disp.requires_grad=True
        inputs.disp.retain_grad()
    else:
        alpha = args.epsilon

    inputs.left.requires_grad=True
    inputs.right.requires_grad=True   
   
    

    # forward pass
    with torch.enable_grad():
        outputs, _,_ = model(inputs)

    # compute loss
        losses = criterion(inputs, outputs)

    if losses is None:
        return outputs, losses, disp
    
    #import ipdb;ipdb.set_trace()
    with torch.enable_grad():
        losses.requires_grad=True
        model.zero_grad()
        #losses.retain_grad()
        #for l in losses:
        #    l.requires_grad=True
        #losses['aggregated'].requires_grad=True

        for i in range(args.iterations):
            #with torch.enable_grad():
            #losses['aggregated'].retain_grad()
        
            #if args.attack == 'fgsm':
            #    losses['aggregated'].backward()
                #print(outputs['disp_pred'].shape)
            if args.attack == 'cospgd':
                cossim= F.cosine_similarity(outputs['disp_pred'], inputs.disp)
                #final_loss = torch.sum(cossim*losses['aggregated']) #/(outputs['disp_pred'].shape[-1]*outputs['disp_pred'].shape[-2])
                final_loss = losses['aggregated']
                with torch.no_grad():
                    final_loss *= torch.sum(cossim)
                #print(outputs['disp_pred'].shape)
                final_loss.backward(retain_graph=True)
            else:
                losses['aggregated'].backward(retain_graph=True)

            #print("iterations: {} \t losses: {}".format(i, losses['aggregated']))
            #logger.info("ATTACKING WITH FGSM")
            #print("before")
            #print(inputs.left.requires_grad)
            #print(inputs.left.grad.mean())
            inputs.left = fgsm_attack(inputs.left, args.epsilon, alpha, inputs.left.grad, left_clip_min, left_clip_max)
            right.left = fgsm_attack(inputs.right, args.epsilon, alpha, inputs.right.grad, right_clip_min, right_clip_max)
            #print("\nafter")
            #print(inputs.left.requires_grad)
            inputs.left.retain_grad() 
            inputs.right.retain_grad()
            if args.attack == 'cospgd':
                inputs.disp.retain_grad()

            #print(inputs.left.grad.mean())

        
            outputs, left_feats, right_feats = model(inputs)
    
            # compute loss
            losses = criterion(inputs, outputs)
            #print("1")
    
    
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

    return outputs, losses, disp, left_feats, right_feats
