import os
import math
import torch
import numpy as np
import numbers
import ipdb


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




class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the super_video should be
    positively labeled, and smaller when the model believes the super_video should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False, all_dataset=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples
        self.all_dataset = all_dataset

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the super_video belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the super_video is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each super_video (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        # Idx of correct preds
        B, C = target.size()
        list_idx_correct_preds = []
        for idx in range(B):
            correct_preds = True
            for j in range(C):
                # does not have the same sign so bad preds -> break
                target_idx_j = -1 if target[idx, j] == 0 else 1
                if target_idx_j * output[idx, j] < 0:
                    correct_preds = False
                    break
            # good preds! if finished the loop
            if correct_preds:
                list_idx_correct_preds.append(idx)

        return list_idx_correct_preds

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]

            # compute average precision
            if self.all_dataset:
                scores_to_keep, targets_to_keep = scores, targets
            else:
                scores_to_keep, targets_to_keep = scores[-100:], targets[-100:]
            ap[k] = AveragePrecisionMeter.average_precision(scores_to_keep, targets_to_keep,
                                                            self.difficult_examples) * 100.
            # ap[k] = AveragePrecisionMeter.compute_ap(scores, targets)

        return ap.mean(), ap.mean(), ap

    @staticmethod
    def compute_ap(scores, targets):
        # import ipdb
        # ipdb.set_trace()
        _, sortind = torch.sort(scores, 0, True)
        truth = targets[sortind]
        rg = torch.range(1, scores.size(0)).float()
        tp = truth.float().cumsum(0)

        # compute precision curve
        precision = tp.div(rg)

        # compute average precision
        ap = precision[truth.byte()].sum() / max(truth.sum(), 1)

        return ap

    # not working
    @staticmethod
    def average_precision(output, target, difficult_examples=True):
        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count + 1e-5
        return precision_at_i

def get_time_to_print(time_sec):
    hours = math.trunc(time_sec / 3600)
    time_sec = time_sec - hours * 3600
    mins = math.trunc(time_sec / 60)
    time_sec = time_sec - mins * 60
    secs = math.trunc(time_sec % 60)
    string = '%02d:%02d:%02d' % (hours, mins, secs)
    return string