from utils.meter import *
import time
import torch
import ipdb
import sys
import torch.nn as nn
from utils.meter import *
import matplotlib
from utils.other import *
import pandas


def make_variable_all_input(dict_input, cuda=False):
    dict_input_var = {}
    for k, v in dict_input.items():
        var = torch.autograd.Variable(v)
        dict_input_var[k] = var.cuda() if cuda else var
    return dict_input_var


def get_obj_id_for_loss(input_var, is_Variable=True, j=0):
    obj_id = input_var['obj_id']
    if is_Variable:
        nb_max_obj = int(torch.max(input_var['max_nb_obj']).cpu())
    else:
        nb_max_obj = int(torch.max(input_var['max_nb_obj'].cpu()))

    # Catch the useful obj id
    obj_id_size = len(obj_id.size())
    obj_id = obj_id[:, :, :nb_max_obj] if obj_id_size == 4 else obj_id[:, :, :, :nb_max_obj]
    obj_id = torch.max(obj_id, -1)[1]
    if obj_id_size == 4:
        obj_id = obj_id.view(-1)
    else:
        obj_id = obj_id[:, j].contiguous()
        obj_id = obj_id.view(-1)

    return obj_id


def forward_backward(model, input_var, criterion, optimizer=None, cuda=False,
                     j=1):
    # compute output
    output = model(input_var)

    # retrieve object id
    obj_id = get_obj_id_for_loss(input_var, j=j)

    # update output
    output = output if isinstance(output, tuple) else (output, None)

    # compute loss
    loss = criterion(output, [input_var['target'], obj_id], cuda)

    # backward
    if optimizer is not None:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  # clip grad
        optimizer.step()

    return output[0], loss  # return the output for the action only - objects preds are just regularizer


def update_metric_loss(input, output, metric, loss, losses):
    # loss
    losses.update(loss.detach(), input['clip'].size(0))

    # metrics
    target = input['target']
    target = target.cpu()
    preds = output.view(-1, output.size(-1)).data.cpu()
    list_idx_correct_preds = metric.add(preds, target)
    metric_val, metric_avg, _ = metric.value()

    return metric_val, metric_avg, list_idx_correct_preds


def take_clip_j(input_var, j):
    input_var_j = {}
    for k, v in input_var.items():
        if k == 'video_id':
            pass
        elif k == 'target':
            input_var_j['target'] = input_var['target']
        elif k == 'id':
            input_var_j['id'] = input_var['id']
        elif k == 'max_nb_obj':
            input_var_j['max_nb_obj'] = input_var['max_nb_obj']
        else:
            input_var_j[k] = input_var[k][:, j]
    return input_var_j


def train(epoch, engine, options, cuda=False):
    # Timer
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Engine
    model = engine['model']
    optimizer = engine['optimizer']
    criterion = engine['criterion']
    metric = engine['metric']()
    data_loader = engine['train_loader']

    # switch to train mode
    model.train()

    end = time.time()
    print("")
    for i, input in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Make Variables
        input_var = make_variable_all_input(input, cuda=cuda)

        # compute output
        output, loss = forward_backward(model, input_var, criterion, optimizer, cuda)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % options['print_freq'] == 0:
            # Do no waste time at computing loss and metric at each iteration of the training process
            metric_val, metric_avg, *_ = update_metric_loss(input, output, metric, loss, losses)

            time_done = get_time_to_print(batch_time.avg * (i + 1))
            time_remaining = get_time_to_print(batch_time.avg * len(data_loader))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) [{done} => {remaining}]\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric {metric_val:.3f} ({metric_avg:.3f})'.format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, metric_val=metric_val,
                metric_avg=metric_avg,
                done=time_done, remaining=time_remaining))
            sys.stdout.flush()

    return losses.avg, metric_avg


def validate(epoch, engine, options, cuda=False):
    # Timer
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # Engine
    model = engine['model']
    criterion = engine['criterion']
    metric = engine['metric']()
    data_loader = engine['val_loader']

    # switch to evaluate mode
    model.eval()

    # create the numpy array for storing the preds and actual target
    dict_id_good_preds, dict_id_failures_preds, dict_id_object = {}, {}, {}

    end = time.time()
    nb_crops = data_loader.dataset.nb_crops
    print("")
    with torch.no_grad():
        for i, input in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # Make Variables
            input_var = make_variable_all_input(input, cuda=cuda)

            output_aggreg, loss_aggreg, obj_id_aggreg = None, None, None
            for j in range(nb_crops):
                # take the right clip
                input_var_j = input_var if nb_crops == 1 else take_clip_j(input_var, j)
                obj_id = input_var_j['obj_id']

                # compute output
                output, loss = forward_backward(model, input_var_j, criterion, None, cuda)
                # ipdb.set_trace()

                # aggreg by summing
                if output_aggreg is None:
                    output_aggreg = output
                    loss_aggreg = loss
                    obj_id_aggreg = obj_id
                else:
                    output_aggreg += output
                    loss_aggreg += loss
                    obj_id_aggreg += obj_id

            # measure accuracy and record loss
            output_aggreg /= nb_crops
            loss_aggreg /= nb_crops
            obj_id_aggreg /= nb_crops
            metric_val, metric_avg, list_idx_correct_preds = update_metric_loss(input, output_aggreg, metric,
                                                                                loss_aggreg, losses)

            # store good and failure cases and detected object
            dict_id_good_i, dict_id_failures_i, dict_obj_i = store_preds(
                output_aggreg, input['id'], list_idx_correct_preds,
                obj_id_aggreg,
                options['dataset'])
            dict_id_good_preds.update(dict_id_good_i)
            dict_id_failures_preds.update(dict_id_failures_i)
            dict_id_object.update(dict_obj_i)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % options['print_freq'] == 0:
                time_done = get_time_to_print(batch_time.avg * (i + 1))
                time_remaining = get_time_to_print(batch_time.avg * len(data_loader))
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) [{done} => {remaining}]\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Metric {metric_val:.3f} ({metric_avg:.3f})'.format(
                    i + 1, len(data_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, metric_val=metric_val,
                    metric_avg=metric_avg,
                    done=time_done, remaining=time_remaining))
                sys.stdout.flush()

    # Finally compute the true mean metric over the overall val set
    metric.all_dataset = True
    _, metric_avg, per_class_metric_avg = metric.value()

    print(' * Metric {metric_avg:.3f}'.format(metric_avg=metric_avg))
    sys.stdout.flush()

    # Pandas frame - good & failure
    df_good = pandas.DataFrame.from_dict(dict_id_good_preds)
    df_failures = pandas.DataFrame.from_dict(dict_id_failures_preds)
    df_objects =  pandas.DataFrame.from_dict(dict_id_object)

    return losses.avg, metric_avg, per_class_metric_avg, df_good, df_failures, df_objects
