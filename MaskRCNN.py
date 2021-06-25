from torch import nn
import torch
import torchvision
from torchvision import models
from torchvision.models.resnet import resnet50
import fpn
import clip
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import loadVOC
from PIL import Image
from torchvision.models.detection.backbone_utils import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
import torchvision.ops.misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from collections import OrderedDict


class maskRCNN(models.detection.faster_rcnn.FasterRCNN):
    def __init__(self, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,):

        backbone = resnet50(pretrained=False)
        state_dict = torch.load("/home/llrt/桌面/DAP/voc_weights_resnet.pth")
        backbone.load_state_dict(state_dict, False)
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:6]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        extra_blocks = LastLevelMaxPool()
        returned_layers = [1, 2, 3, 4]
        return_layers = {f'layer{k}': str(v)
                         for v, k in enumerate(returned_layers)}
        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [in_channels_stage2 *
                            2 ** (i - 1) for i in returned_layers]
        out_channels = 256

        backbone = BackboneWithFPN(
            backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
        super(maskRCNN, self).__init__(backbone, num_classes,
                                       # transform parameters
                                       min_size, max_size,
                                       image_mean, image_std,
                                       # RPN-specific parameters
                                       rpn_anchor_generator, rpn_head,
                                       rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
                                       rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
                                       rpn_nms_thresh,
                                       rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                                       rpn_batch_size_per_image, rpn_positive_fraction,
                                       # Box parameters
                                       box_roi_pool, box_head, box_predictor,
                                       box_score_thresh, box_nms_thresh, box_detections_per_img,
                                       box_fg_iou_thresh, box_bg_iou_thresh,
                                       box_batch_size_per_image, box_positive_fraction,
                                       bbox_reg_weights)
