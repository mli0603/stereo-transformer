#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import argparse
import os
import random

import numpy as np
import torch

from dataset import build_data_loader
from module.sttr import STTR
from utilities.checkpoint_saver import Saver
from utilities.eval import evaluate
from utilities.inference import inference
from utilities.summary_logger import TensorboardSummary
from utilities.train import train_one_epoch
from module.loss import build_criterion

import logging


def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--lr_regression', default=2e-4, type=float)
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    #parser.add_argument('--device', default='cuda',
    #                    help='device to use for training / testing')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--ft', action='store_true', help='load model from checkpoint, but discard optimizer state')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--checkpoint', type=str, default='dev', help='checkpoint name for current experiment')
    parser.add_argument('--pre_train', action='store_true')
    parser.add_argument('--downsample', default=3, type=int,
                        help='This is outdated in STTR-light. Default downsampling is 4 and cannot be changed.')
    parser.add_argument('--apex', action='store_true', help='enable mixed precision training')

    # * STTR
    parser.add_argument('--channel_dim', default=128, type=int,
                        help="Size of the embeddings (dimension of the transformer)")

    # * Positional Encoding
    parser.add_argument('--position_encoding', default='sine1d_rel', type=str, choices=('sine1d_rel', 'none'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--num_attn_layers', default=6, type=int, help="Number of attention layers in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # * Regression Head
    parser.add_argument('--regression_head', default='ot', type=str, choices=('softmax', 'ot'),
                        help='Normalization to be used')
    parser.add_argument('--context_adjustment_layer', default='cal', choices=['cal', 'none'], type=str)
    parser.add_argument('--cal_num_blocks', default=8, type=int)
    parser.add_argument('--cal_feat_dim', default=16, type=int)
    parser.add_argument('--cal_expansion_ratio', default=4, type=int)

    # * Dataset parameters
    parser.add_argument('--dataset', default='sceneflow', type=str, help='dataset to train/eval on')
    parser.add_argument('--dataset_directory', default='', type=str, help='directory to dataset')
    parser.add_argument('--validation', default='validation', type=str, choices={'validation', 'validation_all'},
                        help='If we validate on all provided training images')

    # * Loss
    parser.add_argument('--px_error_threshold', type=int, default=3,
                        help='Number of pixels for error computation (default 3 px)')
    parser.add_argument('--loss_weight', type=str, default='rr:1.0, l1_raw:1.0, l1:1.0, occ_be:1.0',
                        help='Weight for losses')
    parser.add_argument('--validation_max_disp', type=int, default=-1)

    # * Large transposed convolution kernels, plots and FGSM attack
    parser.add_argument('-ks', '--kernel_size', type=int, default=3,
                        help='size of the transposed convolution kernels')
    parser.add_argument('-pk', '--parallel_kernel', action="store_true",
                        help='parallel kernel for large transposed convolution kernels')
    parser.add_argument('-fg', '--fgsm', action="store_true",
                        help='flag for testing against fgsm attack')
    parser.add_argument('-pf', '--plot_freq', action="store_true",
                        help='to get values for plot power vs frequency plots')
    parser.add_argument('-ep', '--epsilon', type=float, default=2.0,
                        help='to pass the epsilon value for FGSM attack')

    return parser


def save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, best, amp=None):
    """
    Save current state of training
    """

    # save model
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_pred': prev_best
    }
    if amp is not None:
        checkpoint['amp'] = amp.state_dict()
    if best:
        checkpoint_saver.save_checkpoint(checkpoint, 'model.pth.tar', write_best=False)
    else:
        checkpoint_saver.save_checkpoint(checkpoint, 'epoch_' + str(epoch) + '_model.pth.tar', write_best=False)


def print_param(model, args):
    """
    print number of parameters in the model
    """

    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad)
    #args.logger2.info('number of params in backbone:', f'{n_parameters:,}')
    args.logger2.info('number of params in backbone: {}'.format(f'{n_parameters:,}'))
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if
                       'transformer' in n and 'regression' not in n and p.requires_grad)
    #args.logger2.info('number of params in transformer:', f'{n_parameters:,}')
    args.logger2.info('number of params in transformer: {}'.format(f'{n_parameters:,}'))
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'tokenizer' in n and p.requires_grad)
    #args.logger2.info('number of params in tokenizer:', f'{n_parameters:,}')
    args.logger2.info('number of params in tokenizer: {}'.format(f'{n_parameters:,}'))
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'regression' in n and p.requires_grad)
    #args.logger2.info('number of params in regression:', f'{n_parameters:,}')
    args.logger2.info('number of params in regression: {}'.format(f'{n_parameters:,}'))


def main(args):
    # get device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # initiate saver and logger
    checkpoint_saver = Saver(args)
    summary_writer = TensorboardSummary(checkpoint_saver.experiment_dir)
    
    os.makedirs(checkpoint_saver.experiment_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_saver.experiment_dir, 'log.log')
    logging.basicConfig(filename=log_file, filemode='a')
    args.logger2 = logging.getLogger("main-logger")
    args.logger2.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    args.logger2.addHandler(handler)

    # build model
    model = STTR(args).to(device)
    #import ipdb;ipdb.set_trace()
    #model = STTR(args).cuda()
    args.logger2.info(model)
    print_param(model, args)

    # set learning rate
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if
                    "backbone" not in n and "regression" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if "regression" in n and p.requires_grad],
            "lr": args.lr_regression,
        },
    ]

    # define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)

    # mixed precision training
    if args.apex:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    else:
        amp = None

    # load checkpoint if provided
    prev_best = np.inf
    if args.resume != '':
        if not os.path.isfile(args.resume):
            raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))

        pretrained_dict = checkpoint['state_dict']
        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
        # check missing and unexpected keys
        if len(missing) > 0:
            args.logger2.info("Missing keys: ", ','.join(missing))
            raise Exception("Missing keys.")
        unexpected_filtered = [k for k in unexpected if
                               'running_mean' not in k and 'running_var' not in k]  # skip bn params
        if len(unexpected_filtered) > 0:
            args.logger2.info("Unexpected keys: ", ','.join(unexpected_filtered))
            raise Exception("Unexpected keys.")
        args.logger2.info("Pre-trained model successfully loaded.")

        # if not ft/inference/eval, load states for optimizer, lr_scheduler, amp and prev best
        if not (args.ft or args.inference or args.eval):
            if len(unexpected) > 0:  # loaded checkpoint has bn parameters, legacy resume, skip loading
                raise Exception("Resuming legacy model with BN parameters. Not possible due to BN param change. " +
                                "Do you want to finetune or inference? If so, check your arguments.")
            else:
                args.start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                prev_best = checkpoint['best_pred']
                if args.apex:
                    amp.load_state_dict(checkpoint['amp'])
                args.logger2.info("Pre-trained optimizer, lr scheduler and stats successfully loaded.")

    # inference
    if args.inference:
        args.logger2.info("Start inference")
        _, _, data_loader = build_data_loader(args)
        inference(model, data_loader, device, args.downsample)

        return    

    # build dataloader
    data_loader_train, data_loader_val, _ = build_data_loader(args)

    # build loss criterion
    criterion = build_criterion(args)

    # eval
    if args.eval:
        args.logger2.info("Start evaluation")
        evaluate(model, criterion, data_loader_val, device, 0, summary_writer, True, output_path=checkpoint_saver.experiment_dir, fgsm=args.fgsm, epsilon=args.epsilon, args=args)
        return

    # train
    args.logger2.info("Start training")
    #import ipdb;ipdb.set_trace()
    for epoch in range(args.start_epoch, args.epochs):
        # train
        args.logger2.info("Epoch: %d" % epoch)
        train_one_epoch(model, data_loader_train, optimizer, criterion, device, epoch, summary_writer,
                        args.clip_max_norm, amp, args)

        # step lr if not pretraining
        if not args.pre_train:
            lr_scheduler.step()
            args.logger2.info("current learning rate: {}".format(lr_scheduler.get_lr()))

        # empty cache
        torch.cuda.empty_cache()

        # save if pretrain, save every 50 epochs
        if args.pre_train or epoch % 50 == 0:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, False, amp)

        # validate
        eval_stats = evaluate(model, criterion, data_loader_val, device, epoch, summary_writer, False, args=args)
        # save if best
        if prev_best > eval_stats['epe'] and 0.5 > eval_stats['px_error_rate']:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, True, amp)

    # save final model
    save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, False, amp)

    return


if __name__ == '__main__':
    ap = argparse.ArgumentParser('STTR training and evaluation script', parents=[get_args_parser()])
    args_ = ap.parse_args()
    main(args_)
