#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-01-11 16:31:57
#   Description : paddlepaddle_yolov3
#
# ================================================================
import cv2
import paddle.fluid as fluid
import paddle.fluid.layers as P
import sys
import time
import shutil
import math
import random
import threading
import numpy as np
import os
from model.darknet_yolo_pd import YOLOv3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def bbox_ciou(boxes1, boxes2):
    '''
    计算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:

    paddle里不支持省略号，boxes1[..., :2]
    '''


    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = P.concat([boxes1[:, :, :, :, :2] - boxes1[:, :, :, :, 2:] * 0.5,
                                boxes1[:, :, :, :, :2] + boxes1[:, :, :, :, 2:] * 0.5], axis=-1)
    boxes2_x0y0x1y1 = P.concat([boxes2[:, :, :, :, :2] - boxes2[:, :, :, :, 2:] * 0.5,
                                boxes2[:, :, :, :, :2] + boxes2[:, :, :, :, 2:] * 0.5], axis=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = P.concat([P.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, :2], boxes1_x0y0x1y1[:, :, :, :, 2:]),
                                P.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, :2], boxes1_x0y0x1y1[:, :, :, :, 2:])], axis=-1)
    boxes2_x0y0x1y1 = P.concat([P.elementwise_min(boxes2_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, 2:]),
                                P.elementwise_max(boxes2_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, 2:])], axis=-1)



    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[:, :, :, :, 2] - boxes1_x0y0x1y1[:, :, :, :, 0]) * (boxes1_x0y0x1y1[:, :, :, :, 3] - boxes1_x0y0x1y1[:, :, :, :, 1])
    boxes2_area = (boxes2_x0y0x1y1[:, :, :, :, 2] - boxes2_x0y0x1y1[:, :, :, :, 0]) * (boxes2_x0y0x1y1[:, :, :, :, 3] - boxes2_x0y0x1y1[:, :, :, :, 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = P.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, :2])
    right_down = P.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, 2:], boxes2_x0y0x1y1[:, :, :, :, 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = P.relu(right_down - left_up)
    inter_area = inter_section[:, :, :, :, 0] * inter_section[:, :, :, :, 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = P.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, :2])
    enclose_right_down = P.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, 2:], boxes2_x0y0x1y1[:, :, :, :, 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = P.pow(enclose_wh[:, :, :, :, 0], 2) + P.pow(enclose_wh[:, :, :, :, 1], 2)

    # 两矩形中心点距离的平方
    p2 = P.pow(boxes1[:, :, :, :, 0] - boxes2[:, :, :, :, 0], 2) + P.pow(boxes1[:, :, :, :, 1] - boxes2[:, :, :, :, 1], 2)

    # 增加av。分母boxes2[:, :, :, :, 3]可能为0，除0保护放在了数据读取阶段preprocess_true_boxes()。
    atan1 = P.atan(boxes1[:, :, :, :, 2] / boxes1[:, :, :, :, 3])
    atan2 = P.atan(boxes2[:, :, :, :, 2] / boxes2[:, :, :, :, 3])
    v = 4.0 * P.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou

def bbox_iou(boxes1, boxes2):
    '''
    预测框          boxes1 (?, grid_h, grid_w, 3,   1, 4)，神经网络的输出(tx, ty, tw, th)经过了后处理求得的(bx, by, bw, bh)
    图片中所有的gt  boxes2 (?,      1,      1, 1, 150, 4)
    paddle里不支持省略号，boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    冒号要写完
    '''
    boxes1_area = boxes1[:, :, :, :, :, 2] * boxes1[:, :, :, :, :, 3]   # 所有格子的3个预测框的面积
    boxes2_area = boxes2[:, :, :, :, :, 2] * boxes2[:, :, :, :, :, 3]   # 所有ground truth的面积

    # (x, y, w, h)变成(x0, y0, x1, y1)
    boxes1 = P.concat([boxes1[:, :, :, :, :, :2] - boxes1[:, :, :, :, :, 2:] * 0.5,
                       boxes1[:, :, :, :, :, :2] + boxes1[:, :, :, :, :, 2:] * 0.5], axis=-1)
    boxes2 = P.concat([boxes2[:, :, :, :, :, :2] - boxes2[:, :, :, :, :, 2:] * 0.5,
                       boxes2[:, :, :, :, :, :2] + boxes2[:, :, :, :, :, 2:] * 0.5], axis=-1)

    # 所有格子的3个预测框 分别 和  150个ground truth  计算iou。 所以left_up和right_down的shape = (?, grid_h, grid_w, 3, 150, 2)
    expand_boxes1 = P.expand(boxes1, [1, 1, 1, 1, P.shape(boxes2)[4], 1])                                      # 不同于pytorch和tf，boxes1和boxes2都要扩展为相同shape
    expand_boxes2 = P.expand(boxes2, [1, P.shape(boxes1)[1], P.shape(boxes1)[2], P.shape(boxes1)[3], 1, 1])    # 不同于pytorch和tf，boxes1和boxes2都要扩展为相同shape
    left_up = P.elementwise_max(expand_boxes1[:, :, :, :, :, :2], expand_boxes2[:, :, :, :, :, :2])  # 相交矩形的左上角坐标
    right_down = P.elementwise_min(expand_boxes1[:, :, :, :, :, 2:], expand_boxes2[:, :, :, :, :, 2:])  # 相交矩形的右下角坐标

    inter_section = P.relu(right_down - left_up)  # 相交矩形的w和h，是负数时取0  (?, grid_h, grid_w, 3, 150, 2)
    inter_area = inter_section[:, :, :, :, :, 0] * inter_section[:, :, :, :, :, 1]  # 相交矩形的面积              (?, grid_h, grid_w, 3, 150)
    expand_boxes1_area = P.expand(boxes1_area, [1, 1, 1, 1, P.shape(boxes2)[4]])
    expand_boxes2_area = P.expand(boxes2_area, [1, P.shape(expand_boxes1_area)[1], P.shape(expand_boxes1_area)[2], P.shape(expand_boxes1_area)[3], 1])
    union_area = expand_boxes1_area + expand_boxes2_area - inter_area  # union_area                (?, grid_h, grid_w, 3, 150)
    iou = 1.0 * inter_area / union_area  # iou                       (?, grid_h, grid_w, 3, 150)

    return iou

def loss_layer(conv, pred, label, bboxes, stride, num_class, iou_loss_thresh, alpha, gamma=2):
    conv_shape = P.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]
    pred_prob = pred[:, :, :, :, 5:]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    ciou = P.reshape(bbox_ciou(pred_xywh, label_xywh), (batch_size, output_size, output_size, 3, 1))    # (8, 13, 13, 3, 1)
    input_size = P.cast(input_size, dtype='float32')

    # 每个预测框xxxiou_loss的权重 = 2 - (ground truth的面积/图片面积)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)   # 1. respond_bbox作为mask，有物体才计算xxxiou_loss

    # 2. respond_bbox作为mask，有物体才计算类别loss
    prob_pos_loss = label_prob * (0 - P.log(pred_prob + 1e-9))             # 二值交叉熵，tf中也是加了极小的常数防止nan
    prob_neg_loss = (1 - label_prob) * (0 - P.log(1 - pred_prob + 1e-9))   # 二值交叉熵，tf中也是加了极小的常数防止nan
    prob_mask = P.expand(respond_bbox, [1, 1, 1, 1, num_class])
    prob_loss = prob_mask * (prob_pos_loss + prob_neg_loss)

    # 3. xxxiou_loss和类别loss比较简单。重要的是conf_loss，是一个focal_loss
    # 分两步：第一步是确定 grid_h * grid_w * 3 个预测框 哪些作为反例；第二步是计算focal_loss。
    expand_pred_xywh = P.reshape(pred_xywh, (batch_size, output_size, output_size, 3, 1, 4))   # 扩展为(?, grid_h, grid_w, 3,   1, 4)
    expand_bboxes = P.reshape(bboxes, (batch_size, 1, 1, 1, P.shape(bboxes)[1], 4))            # 扩展为(?,      1,      1, 1, 150, 4)
    iou = bbox_iou(expand_pred_xywh, expand_bboxes)      # 所有格子的3个预测框 分别 和  150个ground truth  计算iou。   (?, grid_h, grid_w, 3, 150)
    max_iou, max_iou_indices = P.topk(iou, k=1)            # 与150个ground truth的iou中，保留最大那个iou。  (?, grid_h, grid_w, 3, 1)

    # respond_bgd代表  这个分支输出的 grid_h * grid_w * 3 个预测框是否是 反例（背景）
    # label有物体，respond_bgd是0。 没物体的话：如果和某个gt(共150个)的iou超过iou_loss_thresh，respond_bgd是0；如果和所有gt(最多150个)的iou都小于iou_loss_thresh，respond_bgd是1。
    # respond_bgd是0代表有物体，不是反例；  权重respond_bgd是1代表没有物体，是反例。
    # 有趣的是，模型训练时由于不断更新，对于同一张图片，两次预测的 grid_h * grid_w * 3 个预测框（对于这个分支输出）  是不同的。用的是这些预测框来与gt计算iou来确定哪些预测框是反例。
    # 而不是用固定大小（不固定位置）的先验框。
    respond_bgd = (1.0 - respond_bbox) * P.cast(max_iou < iou_loss_thresh, 'float32')

    # RetinaNet的focal_loss，带上alpha解决不平衡问题。
    # pos_loss = respond_bbox * (0 - P.log(pred_conf)) * P.pow(1 - pred_conf, gamma) * alpha
    # neg_loss = respond_bgd  * (0 - P.log(1 - pred_conf)) * P.pow(pred_conf, gamma) * (1 - alpha)

    # 二值交叉熵损失
    pos_loss = respond_bbox * (0 - P.log(pred_conf))
    neg_loss = respond_bgd  * (0 - P.log(1 - pred_conf))

    conf_loss = pos_loss + neg_loss
    # 回顾respond_bgd，某个预测框和某个gt的iou超过iou_loss_thresh，不被当作是反例。在参与“预测的置信位 和 真实置信位 的 二值交叉熵”时，这个框也可能不是正例(label里没标这个框是1的话)。这个框有可能不参与置信度loss的计算。
    # 这种框一般是gt框附近的框，或者是gt框所在格子的另外两个框。它既不是正例也不是反例不参与置信度loss的计算，其实对yolov3算法是有好处的。（论文里称之为ignore）
    # 它如果作为反例参与置信度loss的计算，会降低yolov3的精度。
    # 它如果作为正例参与置信度loss的计算，可能会导致预测的框不准确（因为可能物体的中心都预测不准）。

    ciou_loss = P.reduce_sum(ciou_loss) / batch_size
    conf_loss = P.reduce_sum(conf_loss) / batch_size
    prob_loss = P.reduce_sum(prob_loss) / batch_size

    return ciou_loss + conf_loss + prob_loss

def decode(conv_output, anchors, stride, num_class, grid_offset):
    conv_shape       = P.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    anchor_per_scale = len(anchors)

    conv_output = P.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + num_class))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    pred_xy = (P.sigmoid(conv_raw_dxdy) + grid_offset) * stride
    anchor_t = fluid.layers.assign(np.copy(anchors).astype(np.float32))
    pred_wh = (P.exp(conv_raw_dwdh) * anchor_t) * stride
    pred_xywh = P.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = P.sigmoid(conv_raw_conf)
    pred_prob = P.sigmoid(conv_raw_prob)

    return P.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def yolo_loss(args, num_classes, iou_loss_thresh, anchors, alpha_1, alpha_2, alpha_3):
    conv_lbbox = args[0]   # (?, ?, ?, 3*(num_classes+5))
    conv_mbbox = args[1]   # (?, ?, ?, 3*(num_classes+5))
    conv_sbbox = args[2]   # (?, ?, ?, 3*(num_classes+5))
    label_sbbox = args[3]   # (?, ?, ?, 3, num_classes+5)
    label_mbbox = args[4]   # (?, ?, ?, 3, num_classes+5)
    label_lbbox = args[5]   # (?, ?, ?, 3, num_classes+5)
    true_sbboxes = args[6]   # (?, 150, 4)
    true_mbboxes = args[7]   # (?, 150, 4)
    true_lbboxes = args[8]   # (?, 150, 4)
    label_sbbox_grid_offset = args[9]    # (?, ?, ?, 3, 2)
    label_mbbox_grid_offset = args[10]   # (?, ?, ?, 3, 2)
    label_lbbox_grid_offset = args[11]   # (?, ?, ?, 3, 2)
    pred_sbbox = decode(conv_sbbox, anchors[0], 8, num_classes, label_sbbox_grid_offset)
    pred_mbbox = decode(conv_mbbox, anchors[1], 16, num_classes, label_mbbox_grid_offset)
    pred_lbbox = decode(conv_lbbox, anchors[2], 32, num_classes, label_lbbox_grid_offset)
    loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbboxes, 8, num_classes, iou_loss_thresh, alpha=alpha_1)
    loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbboxes, 16, num_classes, iou_loss_thresh, alpha=alpha_2)
    loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbboxes, 32, num_classes, iou_loss_thresh, alpha=alpha_3)
    return loss_sbbox + loss_mbbox + loss_lbbox

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def training_transform(height, width, output_height, output_width):
    height_scale, width_scale = output_height / height, output_width / width
    scale = min(height_scale, width_scale)
    resize_height, resize_width = round(height * scale), round(width * scale)
    pad_top = (output_height - resize_height) // 2
    pad_left = (output_width - resize_width) // 2
    A = np.float32([[scale, 0.0], [0.0, scale]])
    B = np.float32([[pad_left], [pad_top]])
    M = np.hstack([A, B])
    return M, output_height, output_width

def image_preporcess(image, target_size, gt_boxes=None):
    # 这里改变了一部分原作者的代码。可以发现，传入训练的图片是bgr格式
    ih, iw = target_size
    h, w = image.shape[:2]
    M, h_out, w_out = training_transform(h, w, ih, iw)
    # 填充黑边缩放
    letterbox = cv2.warpAffine(image, M, (w_out, h_out))
    pimage = np.float32(letterbox) / 255.
    if gt_boxes is None:
        return pimage
    else:
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return pimage, gt_boxes

def random_fill(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        # 水平方向填充黑边，以训练小目标检测
        if random.random() < 0.5:
            dx = random.randint(int(0.5*w), int(1.5*w))
            black_1 = np.zeros((h, dx, 3), dtype='uint8')
            black_2 = np.zeros((h, dx, 3), dtype='uint8')
            image = np.concatenate([black_1, image, black_2], axis=1)
            bboxes[:, [0, 2]] += dx
        # 垂直方向填充黑边，以训练小目标检测
        else:
            dy = random.randint(int(0.5*h), int(1.5*h))
            black_1 = np.zeros((dy, w, 3), dtype='uint8')
            black_2 = np.zeros((dy, w, 3), dtype='uint8')
            image = np.concatenate([black_1, image, black_2], axis=0)
            bboxes[:, [1, 3]] += dy
    return image, bboxes

def random_horizontal_flip(image, bboxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0,2]] = w - bboxes[:, [2,0]]
    return image, bboxes

def random_crop(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]
        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))
        image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    return image, bboxes

def random_translate(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]
        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return image, bboxes

def parse_annotation(annotation, train_input_size, annotation_type):
    line = annotation.split()
    image_path = line[0]
    # ss = line[0].split('keras')
    # image_path = '../data/data4379/pascalvoc/'+ss[1]
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = np.array(cv2.imread(image_path))
    # 没有标注物品，即每个格子都当作背景处理
    exist_boxes = True
    if len(line) == 1:
        bboxes = np.array([[10, 10, 101, 103, 0]])
        exist_boxes = False
    else:
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
    if annotation_type == 'train':
        # image, bboxes = random_fill(np.copy(image), np.copy(bboxes))    # 数据集缺乏小物体时打开
        image, bboxes = random_horizontal_flip(np.copy(image), np.copy(bboxes))
        image, bboxes = random_crop(np.copy(image), np.copy(bboxes))
        image, bboxes = random_translate(np.copy(image), np.copy(bboxes))
    image, bboxes = image_preporcess(np.copy(image), [train_input_size, train_input_size], np.copy(bboxes))
    return image, bboxes, exist_boxes

def bbox_iou_data(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return inter_area / union_area

def preprocess_true_boxes(bboxes, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors):
    l1 = np.zeros((train_output_sizes[0], train_output_sizes[0], 3, 5 + num_classes))
    l2 = np.zeros((train_output_sizes[1], train_output_sizes[1], 3, 5 + num_classes))
    l3 = np.zeros((train_output_sizes[2], train_output_sizes[2], 3, 5 + num_classes))
    l1[:, :, :, 3] = 1   # 为了保证后面计算ciou时不出现除0错误
    l2[:, :, :, 3] = 1   # 为了保证后面计算ciou时不出现除0错误
    l3[:, :, :, 3] = 1   # 为了保证后面计算ciou时不出现除0错误
    label = [l1, l2, l3]
    bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(3)]
    bbox_count = np.zeros((3,))
    for bbox in bboxes:
        bbox_coor = bbox[:4]
        bbox_class_ind = bbox[4]
        onehot = np.zeros(num_classes, dtype=np.float)
        onehot[bbox_class_ind] = 1.0
        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]
        iou = []
        for i in range(3):
            anchors_xywh = np.zeros((3, 4))
            anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = anchors[i]
            iou_scale = bbox_iou_data(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
            iou.append(iou_scale)
        best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
        best_detect = int(best_anchor_ind / 3)
        best_anchor = int(best_anchor_ind % 3)
        xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
        # 防止越界
        grid_r = label[best_detect].shape[0]
        grid_c = label[best_detect].shape[1]
        xind = max(0, xind)
        yind = max(0, yind)
        xind = min(xind, grid_r-1)
        yind = min(yind, grid_c-1)
        label[best_detect][yind, xind, best_anchor, :] = 0
        label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
        label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
        label[best_detect][yind, xind, best_anchor, 5:] = onehot
        bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
        bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
        bbox_count[best_detect] += 1
    label_sbbox, label_mbbox, label_lbbox = label
    sbboxes, mbboxes, lbboxes = bboxes_xywh
    return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

def get_grid_offset(grid_n):
    grid_offset = np.arange(grid_n)
    grid_x_offset = np.tile(grid_offset, (grid_n, 1))
    grid_y_offset = np.copy(grid_x_offset)
    grid_y_offset = grid_y_offset.transpose(1, 0)
    grid_x_offset = np.reshape(grid_x_offset, (grid_n, grid_n, 1, 1))
    grid_x_offset = np.tile(grid_x_offset, (1, 1, 3, 1))
    grid_y_offset = np.reshape(grid_y_offset, (grid_n, grid_n, 1, 1))
    grid_y_offset = np.tile(grid_y_offset, (1, 1, 3, 1))
    grid_offset = np.concatenate([grid_x_offset, grid_y_offset], axis=-1)
    return grid_offset

def multi_thread_read(batch, num, train_input_size, annotation_type, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors, batch_image,
                      batch_label_sbbox, batch_label_mbbox, batch_label_lbbox,
                      batch_sbboxes, batch_mbboxes, batch_lbboxes,
                      batch_label_sbbox_grid_offset, batch_label_mbbox_grid_offset, batch_label_lbbox_grid_offset,
                      label_sbbox_grid_offset, label_mbbox_grid_offset, label_lbbox_grid_offset):
    image, bboxes, exist_boxes = parse_annotation(batch[num], train_input_size, annotation_type)
    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = preprocess_true_boxes(bboxes, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors)
    batch_image[num, :, :, :] = image
    if exist_boxes:
        batch_label_sbbox[num, :, :, :, :] = label_sbbox
        batch_label_mbbox[num, :, :, :, :] = label_mbbox
        batch_label_lbbox[num, :, :, :, :] = label_lbbox
        batch_sbboxes[num, :, :] = sbboxes
        batch_mbboxes[num, :, :] = mbboxes
        batch_lbboxes[num, :, :] = lbboxes
        batch_label_sbbox_grid_offset[num, :, :, :, :] = np.copy(label_sbbox_grid_offset)
        batch_label_mbbox_grid_offset[num, :, :, :, :] = np.copy(label_mbbox_grid_offset)
        batch_label_lbbox_grid_offset[num, :, :, :, :] = np.copy(label_lbbox_grid_offset)

def generate_one_batch(annotation_lines, step, batch_size, anchors, num_classes, max_bbox_per_scale, annotation_type):
    n = len(annotation_lines)

    # 多尺度训练
    train_input_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
    train_input_size = random.choice(train_input_sizes)
    strides = np.array([8, 16, 32])

    # 输出的网格数
    train_output_sizes = train_input_size // strides

    batch_image = np.zeros((batch_size, train_input_size, train_input_size, 3))

    batch_label_sbbox = np.zeros((batch_size, train_output_sizes[0], train_output_sizes[0],
                                  3, 5 + num_classes))
    batch_label_mbbox = np.zeros((batch_size, train_output_sizes[1], train_output_sizes[1],
                                  3, 5 + num_classes))
    batch_label_lbbox = np.zeros((batch_size, train_output_sizes[2], train_output_sizes[2],
                                  3, 5 + num_classes))

    batch_sbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
    batch_mbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))
    batch_lbboxes = np.zeros((batch_size, max_bbox_per_scale, 4))

    batch_label_sbbox_grid_offset = np.zeros((batch_size, train_output_sizes[0], train_output_sizes[0], 3, 2))
    batch_label_mbbox_grid_offset = np.zeros((batch_size, train_output_sizes[1], train_output_sizes[1], 3, 2))
    batch_label_lbbox_grid_offset = np.zeros((batch_size, train_output_sizes[2], train_output_sizes[2], 3, 2))

    # 网格偏移
    label_sbbox_grid_offset = get_grid_offset(train_output_sizes[0])
    label_mbbox_grid_offset = get_grid_offset(train_output_sizes[1])
    label_lbbox_grid_offset = get_grid_offset(train_output_sizes[2])

    if (step+1)*batch_size > n:
        batch = annotation_lines[n-batch_size:n]
    else:
        batch = annotation_lines[step*batch_size:(step+1)*batch_size]
    threads = []
    for num in range(batch_size):
        t = threading.Thread(target=multi_thread_read, args=(batch, num, train_input_size, annotation_type, train_output_sizes, strides, num_classes, max_bbox_per_scale, anchors, batch_image,
                      batch_label_sbbox, batch_label_mbbox, batch_label_lbbox,
                      batch_sbboxes, batch_mbboxes, batch_lbboxes,
                      batch_label_sbbox_grid_offset, batch_label_mbbox_grid_offset, batch_label_lbbox_grid_offset,
                      label_sbbox_grid_offset, label_mbbox_grid_offset, label_lbbox_grid_offset))
        threads.append(t)
        t.start()
    # 等待所有线程任务结束。
    for t in threads:
        t.join()
    batch_image = batch_image.transpose(0, 3, 1, 2)
    # paddle里np类型必须和layers.data张量数据类型一致
    batch_image = batch_image.astype(np.float32)
    batch_label_sbbox = batch_label_sbbox.astype(np.float32)
    batch_label_mbbox = batch_label_mbbox.astype(np.float32)
    batch_label_lbbox = batch_label_lbbox.astype(np.float32)
    batch_sbboxes = batch_sbboxes.astype(np.float32)
    batch_mbboxes = batch_mbboxes.astype(np.float32)
    batch_lbboxes = batch_lbboxes.astype(np.float32)
    batch_label_sbbox_grid_offset = batch_label_sbbox_grid_offset.astype(np.float32)
    batch_label_mbbox_grid_offset = batch_label_mbbox_grid_offset.astype(np.float32)
    batch_label_lbbox_grid_offset = batch_label_lbbox_grid_offset.astype(np.float32)
    return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes, batch_label_sbbox_grid_offset, batch_label_mbbox_grid_offset, batch_label_lbbox_grid_offset

if __name__ == '__main__':
    train_path = 'annotation/voc2012_train.txt'
    val_path = 'annotation/voc2012_val.txt'
    classes_path = 'data/voc_classes.txt'

    # train_path = 'annotation/coco2017_train.txt'
    # val_path = 'annotation/coco2017_val.txt'
    # classes_path = 'data/coco_classes.txt'

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = np.array([
        [[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]],
        [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],
        [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]
    ])

    # 模式。 0-从头训练，1-读取模型继续训练（包括解冻），2-读取coco预训练模型训练
    pattern = 0
    is_test = False
    use_gpu = True
    # use_gpu = False

    max_bbox_per_scale = 150
    iou_loss_thresh = 0.7

    # 经过试验发现，使用focal_loss会增加误判fp，所以默认使用二值交叉熵损失函数训练。下面这3个alpha请忽略。
    # 经过试验发现alpha取>0.5的值时mAP会提高，但误判（False Predictions）会增加；alpha取<0.5的值时mAP会降低，误判会降低。
    # 试验时alpha_1取0.95，alpha_2取0.85，alpha_3取0.75
    # 小感受野输出层输出的格子最多，预测框最多，正样本很有可能占比是最少的，所以试验时alpha_1 > alpha_2 > alpha_3
    alpha_1 = 0.5    # 小感受野输出层的focal_loss的alpha
    alpha_2 = 0.5    # 中感受野输出层的focal_loss的alpha
    alpha_3 = 0.5    # 大感受野输出层的focal_loss的alpha

    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = None
    with fluid.program_guard(train_program, startup_program):
        # 多尺度训练
        inputs = P.data(name='input_1', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
        if pattern == 2:
            lr = 0.0001
            batch_size = 24
            initial_epoch = 0
            epochs = 130
            initial_filters = 32
            model_path = 'aaa'

            y1, y2, y3 = YOLOv3(inputs, initial_filters, num_classes, is_test, trainable=False)
        elif pattern == 1:
            lr = 0.0001
            batch_size = 24
            initial_epoch = 65
            epochs = 130
            initial_filters = 32
            model_path = 'ep000002-loss8468.868-val_loss8757.135.pd'

            y1, y2, y3 = YOLOv3(inputs, initial_filters, num_classes, is_test)
        elif pattern == 0:
            lr = 0.0001
            batch_size = 24
            initial_epoch = 0
            epochs = 130
            initial_filters = 32

            y1, y2, y3 = YOLOv3(inputs, initial_filters, num_classes, is_test)
        # 建立损失函数
        label_sbbox = P.data(name='input_2', shape=[-1, -1, -1, 3, (num_classes + 5)], append_batch_size=False, dtype='float32')
        label_mbbox = P.data(name='input_3', shape=[-1, -1, -1, 3, (num_classes + 5)], append_batch_size=False, dtype='float32')
        label_lbbox = P.data(name='input_4', shape=[-1, -1, -1, 3, (num_classes + 5)], append_batch_size=False, dtype='float32')
        true_sbboxes = P.data(name='input_5', shape=[max_bbox_per_scale, 4], dtype='float32')
        true_mbboxes = P.data(name='input_6', shape=[max_bbox_per_scale, 4], dtype='float32')
        true_lbboxes = P.data(name='input_7', shape=[max_bbox_per_scale, 4], dtype='float32')
        label_sbbox_grid_offset = P.data(name='input_8', shape=[-1, -1, -1, 3, 2], append_batch_size=False, dtype='float32')
        label_mbbox_grid_offset = P.data(name='input_9', shape=[-1, -1, -1, 3, 2], append_batch_size=False, dtype='float32')
        label_lbbox_grid_offset = P.data(name='input_10', shape=[-1, -1, -1, 3, 2], append_batch_size=False, dtype='float32')
        args = [y1, y2, y3, label_sbbox, label_mbbox, label_lbbox, true_sbboxes, true_mbboxes, true_lbboxes, label_sbbox_grid_offset, label_mbbox_grid_offset, label_lbbox_grid_offset]
        loss = yolo_loss(args, num_classes, iou_loss_thresh, anchors, alpha_1, alpha_2, alpha_3)
        loss.persistable = True

        # 在使用Optimizer之前，将train_program复制成一个test_program。之后使用测试数据运行test_program，就可以做到运行测试程序，而不影响训练结果。
        test_program = train_program.clone(for_test=True)

        # 写完网络和损失，要紧接着写优化器
        optimizer = fluid.optimizer.Adam(learning_rate=lr)
        optimizer.minimize(loss)

    # 参数随机初始化
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)

    if pattern == 1:
        fluid.io.load_persistables(exe, model_path, main_program=startup_program)

    # 验证集和训练集
    with open(train_path) as f:
        train_lines = f.readlines()
    with open(val_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 一轮的步数
    train_steps = int(num_train / batch_size) if num_train % batch_size == 0 else int(num_train / batch_size)+1
    val_steps = int(num_val / batch_size) if num_val % batch_size == 0 else int(num_val / batch_size)+1

    best_val_loss = 0.0
    for t in range(initial_epoch, epochs, 1):
        print('Epoch %d/%d\n'%(t+1, epochs))
        epochStartTime = time.time()
        start = time.time()
        # 每个epoch之前洗乱
        np.random.shuffle(train_lines)
        train_epoch_loss, val_epoch_loss = [], []

        # 训练阶段
        for step in range(train_steps):
            batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes, batch_label_sbbox_grid_offset, batch_label_mbbox_grid_offset, batch_label_lbbox_grid_offset = generate_one_batch(train_lines, step, batch_size, anchors, num_classes, max_bbox_per_scale, 'train')

            train_step_loss, = exe.run(train_program, feed={"input_1": batch_image, "input_2": batch_label_sbbox,
                                                            "input_3": batch_label_mbbox, "input_4": batch_label_lbbox,
                                                            "input_5": batch_sbboxes, "input_6": batch_mbboxes,
                                                            "input_7": batch_lbboxes, "input_8": batch_label_sbbox_grid_offset,
                                                            "input_9": batch_label_mbbox_grid_offset, "input_10": batch_label_lbbox_grid_offset}, fetch_list=[loss.name])
            train_epoch_loss.append(train_step_loss)

            # 自定义进度条
            percent = ((step + 1) / train_steps) * 100
            num = int(29 * percent / 100)
            ETA = int((time.time() - epochStartTime) * (100 - percent) / percent)
            sys.stdout.write('\r{0}'.format(' ' * (len(str(train_steps)) - len(str(step + 1)))) + \
                             '{0}/{1} [{2}>'.format(step + 1, train_steps, '=' * num) + '{0}'.format('.' * (29 - num)) + ']' + \
                             ' - ETA: ' + str(ETA) + 's' + ' - loss: %.4f'%(train_step_loss, ))
            sys.stdout.flush()

        # 验证阶段前后参数没有发生变化
        # aaaa11 = np.array(fluid.global_scope().find_var('conv59.conv.weights').get_tensor())
        # aaaa12 = np.array(fluid.global_scope().find_var('conv67.conv.weights').get_tensor())
        # aaaa13 = np.array(fluid.global_scope().find_var('conv75.conv.weights').get_tensor())
        # aaaa14 = np.array(fluid.global_scope().find_var('conv74.bn.scale').get_tensor())
        # aaaa15 = np.array(fluid.global_scope().find_var('conv74.bn.offset').get_tensor())
        # aaaa16 = np.array(fluid.global_scope().find_var('conv74.bn.mean').get_tensor())
        # aaaa17 = np.array(fluid.global_scope().find_var('conv74.bn.var').get_tensor())
        # 验证阶段
        for step in range(val_steps):
            batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes, batch_label_sbbox_grid_offset, batch_label_mbbox_grid_offset, batch_label_lbbox_grid_offset = generate_one_batch(val_lines, step, batch_size, anchors, num_classes, max_bbox_per_scale, 'val')

            val_step_loss, = exe.run(test_program, feed={"input_1": batch_image, "input_2": batch_label_sbbox,
                                                         "input_3": batch_label_mbbox, "input_4": batch_label_lbbox,
                                                         "input_5": batch_sbboxes, "input_6": batch_mbboxes,
                                                         "input_7": batch_lbboxes, "input_8": batch_label_sbbox_grid_offset,
                                                         "input_9": batch_label_mbbox_grid_offset, "input_10": batch_label_lbbox_grid_offset}, fetch_list=[loss.name])
            val_epoch_loss.append(val_step_loss)
        train_epoch_loss, val_epoch_loss = np.mean(train_epoch_loss), np.mean(val_epoch_loss)
        # 验证阶段前后参数没有发生变化
        # aaaa21 = np.array(fluid.global_scope().find_var('conv59.conv.weights').get_tensor())
        # aaaa22 = np.array(fluid.global_scope().find_var('conv67.conv.weights').get_tensor())
        # aaaa23 = np.array(fluid.global_scope().find_var('conv75.conv.weights').get_tensor())
        # aaaa24 = np.array(fluid.global_scope().find_var('conv74.bn.scale').get_tensor())
        # aaaa25 = np.array(fluid.global_scope().find_var('conv74.bn.offset').get_tensor())
        # aaaa26 = np.array(fluid.global_scope().find_var('conv74.bn.mean').get_tensor())
        # aaaa27 = np.array(fluid.global_scope().find_var('conv74.bn.var').get_tensor())

        # 保存模型
        content = '%d\tloss = %.4f\tval_loss = %.4f\n' % ((t + 1), train_epoch_loss, val_epoch_loss)
        with open('yolov3_paddle_logs.txt', 'a', encoding='utf-8') as f:
            f.write(content)
            f.close()
        path_dir = os.listdir('./')
        eps = []
        names = []
        for name in path_dir:
            if name[len(name) - 2:len(name)] == 'pd' and name[0:2] == 'ep':
                sss = name.split('-')
                ep = int(sss[0][2:])
                eps.append(ep)
                names.append(name)
        if len(eps) >= 10:
            i2 = eps.index(min(eps))
            shutil.rmtree(names[i2])
        best_val_loss = val_epoch_loss
        fluid.io.save_persistables(exe, 'ep%.6d-loss%.3f-val_loss%.3f.pd' % ((t + 1), train_epoch_loss, val_epoch_loss), train_program)

        # 打印本轮训练结果
        sys.stdout.write(
            '\r{0}/{1} [{2}='.format(train_steps, train_steps, '=' * num) + '{0}'.format('.' * (29 - num)) + ']' + \
            ' - %ds' % (int(time.time() - epochStartTime),) + ' - loss: %.4f'%(train_epoch_loss, ) + ' - val_loss: %.4f\n'%(val_epoch_loss, ))
        sys.stdout.flush()

