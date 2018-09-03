import torch.utils.model_zoo as model_zoo
from model.backbone.resnet.resnet import model_urls
import ipdb
import torch


def _inflate_weight(w, new_temporal_size, inflation='center'):
    w_up = w.unsqueeze(2).repeat(1, 1, new_temporal_size, 1, 1)
    if inflation == 'center':
        w_up = central_inflate_3D_conv(w_up)  # center
    elif inflation == 'mean':
        w_up /= new_temporal_size  # mean
    return w_up


def central_inflate_3D_conv(w):
    new_temporal_size = w.size(2)
    middle_timestep = int(new_temporal_size / 2.)
    before, after = list(range(middle_timestep)), list(range(middle_timestep + 1, new_temporal_size))
    if len(before) > 0:
        w[:, :, before] = torch.zeros_like(w[:, :, before])
    if len(after):
        w[:, :, after] = torch.zeros_like(w[:, :, after])
    return w


def inflate_temporal_conv(pretrained_W_updated, model_dict, inflation):
    for k, v in model_dict.items():
        if '1t' in k:
            if 'conv' in k:
                if 'bias' in k:
                    v_up = torch.zeros_like(v)
                elif 'weight' in k:
                    v_up = torch.zeros_like(v)

                    h, w, T, *_ = v_up.size()
                    t_2 = int(T / 2.)
                    for i in range(h):
                        for j in range(w):
                            if i == j:
                                if inflation == 'center':
                                    v_up[i, j, t_2] = torch.ones_like(v_up[i, j, t_2])
                                elif inflation == 'mean':
                                    v_up[i, j] = torch.ones_like(v_up[i, j]) / T
            # elif 'bn' in k:
            #     if 'running_mean' in k:
            #         v_up = torch.zeros_like(v)
            #     elif 'running_var' in k:
            #         v_up = torch.ones_like(v) - 1e-05
            #     elif 'bias' in k:
            #         v_up = torch.zeros_like(v)
            #     elif 'weight' in k:
            #         v_up = torch.ones_like(v)

                # udpate
                pretrained_W_updated.update({k: v_up})
    return pretrained_W_updated


def _update_pretrained_weights(model, pretrained_W, inflation='center'):
    pretrained_W_updated = pretrained_W.copy()
    model_dict = model.state_dict()
    for k, v in pretrained_W.items():
        if "conv" in k or ('bn' in k and '1t' not in k) or 'downsample' in k:
            if k in model_dict.keys():
                if len(model_dict[k].shape) == 5:
                    new_temporal_size = model_dict[k].size(2)
                    v_updated = _inflate_weight(v, new_temporal_size, inflation)
                else:
                    v_updated = v

                if isinstance(v, torch.autograd.Variable):
                    pretrained_W_updated.update({k: v_updated.data})
                else:
                    pretrained_W_updated.update({k: v_updated})
        if "fc.weight" in k:
            pretrained_W_updated.pop('fc.weight', None)
        if "fc.bias" in k:
            pretrained_W_updated.pop('fc.bias', None)

    # update the dict for 1D conv for 2.5D conv
    pretrained_W_updated = inflate_temporal_conv(pretrained_W_updated, model_dict, inflation)

    # update the state dict
    model_dict.update(pretrained_W_updated)

    return model_dict


def _keep_only_existing_keys(model, pretrained_weights_inflated):
    # Loop over the model_dict and update W
    model_dict = model.state_dict()  # Take the initial weights
    for k, v in model_dict.items():
        if k in pretrained_weights_inflated.keys():
            model_dict[k] = pretrained_weights_inflated[k]
    return model_dict


def load_pretrained_2D_weights(arch, model, inflation):
    pretrained_weights = model_zoo.load_url(model_urls[arch])
    pretrained_weights_inflated = _update_pretrained_weights(model, pretrained_weights, inflation)
    model.load_state_dict(pretrained_weights_inflated)
    print("     -> Init: Imagenet - 3D from 2D (inflation = {})".format(inflation))
    return model
