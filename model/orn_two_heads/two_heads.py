from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from model.backbone.resnet_based import resnet_two_heads
import numpy as np
import random
from model.orn_two_heads.encoder import EncoderMLP
from model.orn_two_heads.classifier import Classifier
from model.backbone.resnet.bottleneck import Bottleneck2D, Bottleneck3D, Bottleneck2_1D
from model.backbone.resnet.basicblock import BasicBlock2D, BasicBlock3D, BasicBlock2_1D
from model.orn_two_heads.aggregation_relations import AggregationRelations
import math
from model.orn_two_heads.orn import ObjectRelationNetwork
import ipdb

__all__ = [
    'orn_two_heads',
]


class TwoHeads(nn.Module):
    def __init__(self, num_classes=174, cnn=None,
                 features_size=512, time=8, mask_size=28, size_RN=256, nb_head=2,
                 logits_type='object', f_orn=True,
                 size_2nd_head=14,
                 **kwargs):
        super(TwoHeads, self).__init__()
        # Basic settings
        self.num_classes = num_classes
        self.time = time
        self.size_cnn_features = features_size
        self.size_mask = mask_size
        self.size_RN = size_RN
        self.size_RNN = size_RN
        self.nb_head = nb_head
        self.logits_type = logits_type
        self.size_fm_2nd_head = size_2nd_head

        # CNN: 4 first conv are shared and the 5th blocks is split
        self.cnn = cnn
        self.cnn.out_dim = 5

        # # If object head only then freeze the cnn because it has already trained for the object recognition task
        if self.logits_type == 'object':
            # freeze the 2 first blocks and the object head
            for i, child in enumerate(self.cnn.children()):
                if i != 7:
                    for param in child.parameters():
                        param.requires_grad = False

        # Average Pooling
        self.avgpool_TxMxM = nn.AvgPool3d((self.time, self.size_fm_2nd_head, self.size_fm_2nd_head))
        self.avgpool_1xMxM = nn.AvgPool3d((1, self.size_fm_2nd_head, self.size_fm_2nd_head))
        self.avgpool_1_7x7 = nn.AvgPool3d((1, 7, 7))  # for pooling features from the context head
        self.avgpool_T_7x7 = nn.AvgPool3d((self.time, 7, 7))  # for pooling features from the context head
        self.pixel_pooler = nn.AvgPool3d((1, 2, 2)) if self.size_fm_2nd_head == 14 else  nn.AvgPool3d(
            (1, 4, 4))  # from 14x14 to 7x7

        # Max pooling for the ORN module
        self.pool_orn = nn.MaxPool2d((self.time - 1, 1))

        # COCO object features
        self.size_COCO_object_features = self.size_cnn_features

        # Prediction of the class of each detected COCO objects (MLP from the pooled features)
        self.COCO_Object_Class_from_Features = Classifier(
            size_input=self.size_cnn_features,
            size_output=81)

        # Embedding of the binary mask by AutoEncoder
        # Goal -> find the latent space of the shape and location of the objects
        self.size_mask_embedding = 100
        self.Encoder_Binary_Mask = EncoderMLP(input_size=self.size_mask * self.size_mask,
                                              list_hidden_size=[self.size_mask_embedding, self.size_mask_embedding])

        # Embedding of the object id
        # Goal -> find the latent space of object id better than just a one hot vector
        self.size_obj_embedding = 100
        input_size = 81
        self.Encoder_COCO_Obj_Class = EncoderMLP(input_size=input_size,
                                                 list_hidden_size=[self.size_obj_embedding, self.size_obj_embedding])

        # Object Relational Network (ORN) between coco or pixel objects
        size_object = self.size_COCO_object_features + self.size_mask_embedding + self.size_obj_embedding

        # ORN
        list_size_hidden_layers = [self.size_RN, self.size_RN, self.size_RN]
        self.ORN = ObjectRelationNetwork(size_object=size_object,
                                         list_hidden_layers_size=list_size_hidden_layers
                                         )
        # Aggregation over the
        self.AggregationRelations = AggregationRelations()

        # RNN or AVG
        self.f_orn = f_orn
        # RNN Object
        if self.f_orn == 'rnn':
            input_size = self.size_RN
            self.size_relation_features = int(self.size_RN / 2.)
            self.rnn_objects = nn.GRU(input_size=input_size,
                                      hidden_size=self.size_relation_features,
                                      num_layers=1,
                                      batch_first=True)
        else:
            self.size_relation_features = self.size_RN

        ## Final classification
        self.fc_classifier_object = nn.Linear(self.size_relation_features, self.num_classes)
        self.fc_classifier_context = nn.Linear(self.size_cnn_features, self.num_classes)

    def get_objects_features(self, fm, masks):
        # Upsample the features maps to get better precision!
        # fm_old = fm
        fm = fm.transpose(1, 2)  # (B, T, D, 14, 14)
        B, T, D, W, H = fm.size()
        fm = fm.view(B * T, D, W, H)
        fm_up = F.upsample(fm, size=(self.size_mask, self.size_mask), mode='bilinear', align_corners=True)
        fm_up = fm_up.view(B, T, D, self.size_mask, self.size_mask)
        fm_up = fm_up.transpose(1, 2)  # (B, D, T, 28, 28)

        # B and K
        B, D, T, W, _ = fm_up.size()
        K = masks.size(2)
        W_masks = masks.size(-1)

        # Make it as the same size and do Hadamard product
        fm_plus = fm_up.unsqueeze(1)  # (B,1,D,T,7,7)
        masks_plus = masks.transpose(1, 2)
        masks_plus = masks_plus.unsqueeze(2)  # (B,K,1,T,7,7)
        fm_masked = fm_plus * masks_plus

        # Area of the objects in the "image"
        masks_size = torch.sum(torch.sum(masks_plus, -1), -1)  # (B,K,1,T)

        list_object_set = []
        for t in range(T):
            # Pool the objects features
            object_set_avg = torch.sum(torch.sum(fm_masked[:, :, :, t], -1), -1)
            object_set_avg /= (masks_size[:, :, :, t] + 1e-4)

            # Append
            list_object_set.append(object_set_avg)

        # Stack
        objects_features = torch.stack(list_object_set, 1)  # (B,T,K,D)

        return objects_features

    def get_pixel_features(self, fm):
        *_, W, H = fm.size()
        fm = fm.transpose(1, 2)
        list_pixel_features = []

        if W != 7:
            fm = self.pixel_pooler(fm)
            *_, W, H = fm.size()

        for i in range(W):
            for j in range(H):
                list_pixel_features.append(fm[:, :, :, i, j])
        pixel_features = torch.stack(list_pixel_features, 2)

        return pixel_features

    def retrieve_relations(self, relational_reasoning_vector_COCO, obj_id):
        B, T, K, C = obj_id.size()
        B, T_1, K2, D = relational_reasoning_vector_COCO.size()

        relations = np.zeros((B, 81, 81))
        for b in range(B):
            for t in range(T_1):
                for k2 in range(K2):
                    try:
                        # k2-th interaction
                        inter = torch.sum(relational_reasoning_vector_COCO[b, t, k2])

                        ## Find the corresponding objects
                        # k_1 object of previous timestep
                        k_1 = math.floor(float(k2) / float(K))
                        obj_id_k_1 = self.get_id_object(obj_id[b, t, k_1].data.cpu().numpy())
                        # k object of current timestep
                        k = k2 - k_1 * K
                        obj_id_k = self.get_id_object(obj_id[b, t + 1, k].data.cpu().numpy())

                        # Add the relation
                        relations[b, obj_id_k_1, obj_id_k] = inter  # matrix but we fill only half of it (the triangle)
                    except:
                        pass
                        # import ipdb
                        # ipdb.set_trace()
        return relations  # (B, 81, 81)

    def retrieve_relations_temporal(self, relational_reasoning_vector_COCO, obj_id):
        B, T, K, C = obj_id.size()
        B, T_1, K2, D = relational_reasoning_vector_COCO.size()

        relations = np.zeros((B, T_1, 81, 81))
        for b in range(B):
            for t in range(T_1):
                for k2 in range(K2):
                    try:
                        # k2-th interaction
                        inter = torch.sum(relational_reasoning_vector_COCO[b, t, k2])

                        ## Find the corresponding objects
                        # k_1 object of previous timestep
                        k_1 = math.floor(float(k2) / float(K))
                        obj_id_k_1 = self.get_id_object(obj_id[b, t, k_1].data.cpu().numpy())
                        # k object of current timestep
                        k = k2 - k_1 * K
                        obj_id_k = self.get_id_object(obj_id[b, t + 1, k].data.cpu().numpy())

                        # Add the relation
                        relations[
                            b, t, obj_id_k_1, obj_id_k] = inter  # matrix but we fill only half of it (the triangle)
                    except:
                        import ipdb
                        pass
                        # ipdb.set_trace()
        return relations  # (B, T-1, 81, 81)

    @staticmethod
    def get_id_object(one_hot):
        if one_hot[0] == 1:
            return 0
        else:
            return np.argmax(one_hot)

    def context_head(self, fm_context, B):
        # 3D GAP
        context_vector = self.avgpool_T_7x7(fm_context)
        context_representation = context_vector.view(B, self.size_cnn_features)

        return context_representation

    def object_head(self, fm_objects, masks, obj_id, B):
        # Retrieve the feature vector associated to each detected COCO object
        objects_features = self.get_objects_features(fm_objects, masks)

        # Classify each detected objects to make sure we extract good object descriptors
        preds_class_detected_objects = self.COCO_Object_Class_from_Features(objects_features)

        # Reconstruct the binary masks to find the correct embedding (i.e. shape and location of the detected objects)
        embedding_objects_location = self.Encoder_Binary_Mask(masks)

        # Reconstruct the COCO class id given the one hot vector
        embedding_obj_id = self.Encoder_COCO_Obj_Class(obj_id)

        # Full objects description
        full_objects = torch.cat([objects_features, embedding_objects_location, embedding_obj_id],
                                 -1)  # (B,T,K,object_size)

        # Run the Relational Reasoning over the different set of COCO objects
        D = self.size_cnn_features  # 512
        all_e, all_is_obj = self.ORN(full_objects, D)  # [B, T-1, K*K, D]

        # Get only interactions where at least one obj is involved (for COCO only)
        all_is_obj = all_is_obj.unsqueeze(-1)
        all_e *= all_is_obj

        # Aggregation of the COCO relations
        orn_aggregated = self.AggregationRelations(all_e)

        if self.f_orn == 'rnn':
            # self.rnn_objects.flatten_parameters() # does not work for multi-GPU ---> bug https://github.com/pytorch/pytorch/issues/7092
            object_representation, _ = self.rnn_objects(orn_aggregated)
            # self.rnn_objects.flatten_parameters()
            object_representation = torch.mean(object_representation,
                                               1)  # TODO look at the two lines above for a better pooling over time!
            # object_representation = self.pool_orn(object_representation) # TODO interesting point for pooling over the hidden states
        elif self.f_orn == 'avg':
            object_representation = torch.mean(orn_aggregated, 1)
        else:
            raise Exception

        return object_representation, preds_class_detected_objects

    def add(self, logits, logits_head):
        if logits is None:
            return logits_head
        else:
            return logits + logits_head

    def squeeze_masks(self, max_nb_obj, masks, obj_id, bbox):
        # Max number of objects
        nb_max_obj_in_B = int(torch.max(max_nb_obj).cpu().numpy())

        # Squeeze
        masks_squeezed = masks[:, :, :nb_max_obj_in_B]
        obj_id_squeezed = obj_id[:, :, :nb_max_obj_in_B]
        bbox_squeezed = bbox[:, :, :nb_max_obj_in_B]

        return masks_squeezed, obj_id_squeezed, bbox_squeezed

    def final_classification(self, context_representation, object_representation):
        if context_representation is not None:
            return self.fc_classifier_object(object_representation) + self.fc_classifier_context(context_representation)
        else:
            return self.fc_classifier_object(object_representation)

    def forward(self, x):
        """Forward pass from a tensor of size (B,C,T,W,H)"""
        clip, masks, obj_id, bbox, max_nb_obj = x['clip'], x['mask'], x['obj_id'], x['obj_bbox'], x['max_nb_obj']

        # Get only the real detected objects
        masks, obj_id, bbox = self.squeeze_masks(max_nb_obj, masks, obj_id, bbox)

        # Get the batch size and the temporal dimension
        B = clip.size(0)  # batch size
        T = clip.size(2)  # number of timesteps in the sequence
        K = masks.size(2)  # number of objects

        # Get the two feature maps: context and object reasoning
        fm_context, fm_objects = self.cnn.get_two_heads_feature_maps(x, T=T, out_dim=5,
                                                                     heads_type=self.logits_type)

        # Init returned variable
        context_representation, object_representation, preds_class_detected_objects = None, None, None

        # HEADS
        if 'object' in self.logits_type:
            object_representation, preds_class_detected_objects = self.object_head(fm_objects, masks, obj_id, B=B)

        if 'context' in self.logits_type:
            context_representation = self.context_head(fm_context, B=B)

        # Final classification
        logits = self.final_classification(context_representation, object_representation)

        return logits, preds_class_detected_objects


def orn_two_heads(options, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    # Settings
    depth, pooling, heads, mask_size, size_2nd_head, time = options['depth'], \
                                                            options['pooling'], \
                                                            options['heads'], \
                                                            14, \
                                                            14, \
                                                            options['t']

    print("* TWO-HEADS => Object type: {} , F: {}, Heads: {}".format('coco', pooling, heads))

    # CNN
    cnn = resnet_two_heads(options, **kwargs)

    # Features dim
    features_size = 2048 if depth > 34 else 512
    size_RN = 512 if depth > 34 else 256

    # Model
    model = TwoHeads(
        cnn=cnn,
        features_size=features_size,
        size_RN=size_RN,
        logits_type=heads,
        f_orn=pooling,
        mask_size=28,
        size_2nd_head=14,
        time=time,
        **kwargs)

    return model
