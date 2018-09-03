from model import models
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.meter import *
from inference.train_val import *
import ipdb
from model import models
from utils.other import *


def main(options):
    # CUDA
    cuda = torch.cuda.is_available()

    # Dataset
    train_dataset, val_dataset, train_loader, val_loader = get_datasets_and_dataloaders(options, cuda=cuda)
    print('\n*** Train set of size {}  -  Val set of size {} ***\n'.format(print_number(len(train_dataset)),
                                                                           print_number(len(val_dataset))))

    # Model
    model = models.__dict__[options['arch']](num_classes=train_dataset.nb_classes,
                                             size_fm_2nd_head=train_dataset.h_mask,
                                             options=options)
    model = model.cuda() if cuda else model
    model = torch.nn.DataParallel(model)

    # Trainable params
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Print number of parameters
    nb_total_params = count_nb_params(model.parameters())
    nb_trainable_params = count_nb_params(filter(lambda p: p.requires_grad, model.parameters()))
    ratio = float(nb_trainable_params / nb_total_params) * 100.
    print("\n* Parameter numbers : {} ({}) - {ratio:.2f}% of the weights are trainable".format(
        print_number(nb_total_params),
        print_number(nb_trainable_params),
        ratio=ratio
    ))

    # Optimizer
    optimizer = torch.optim.Adam(trainable_params, options['lr'])

    # Loss function and Metric
    criterion, metric = get_loss_and_metric(options)

    # Load resume from resume if exists
    model, optimizer, epoch = load_from_dir(model, optimizer, options)

    # My engine
    engine = {'model': model,
              'optimizer': optimizer, 'criterion': criterion, 'metric': metric,
              'train_loader': train_loader, 'val_loader': val_loader}

    # Training/Val or Testing #
    if options['evaluate']:
        # Val
        loss_val, metric_val, per_class_metric_val, df_good, df_failure, df_objects = validate(epoch, engine, options, cuda=cuda)
        # Write into log
        write_to_log(val_dataset.dataset, options['resume'], epoch, [loss_val, metric_val], per_class_metric_val)
        # Save good and failures and object presence
        df_good.to_csv(os.path.join(options['resume'], 'df_good_preds.csv'), sep=',', encoding='utf-8')
        df_failure.to_csv(os.path.join(options['resume'], 'df_failure_preds.csv'), sep=',', encoding='utf-8')
        df_objects.to_csv(os.path.join(options['resume'], 'df_objects'), sep=',', encoding='utf-8')

    else:
        # Train (and Val if having access tto the val set)
        is_best = True
        best_metric_val = -0.1
        for epoch in range(1, options['epochs'] + 1):
            # train one epoch
            loss_train, metric_train = train(epoch, engine, options, cuda=cuda)

            # write into log
            write_to_log(train_dataset.dataset, options['resume'], epoch, [loss_train, loss_train], None)

            # get the val metric
            if options['train_set'] == 'train':
                # Val
                loss_val, metric_val, per_class_metric_val, *_ = validate(epoch, engine, options, cuda=cuda)
                # Write into log
                write_to_log(val_dataset.dataset, options['resume'], epoch, [loss_val, metric_val],
                             per_class_metric_val)

                # Best compared to previous checkpoint ?
                is_best = metric_val > best_metric_val
                best_metric_val = max(metric_val, best_metric_val)

            # save checkpoint
            save_checkpoint({
                'epoch': epoch,
                'arch': options['arch'],
                'state_dict': model.state_dict(),
                'best_metric_val': best_metric_val,
                'optimizer': optimizer.state_dict(),
            }, is_best, options['resume'])

    return None
