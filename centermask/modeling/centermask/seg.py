from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class FeatureUp(nn.Module):
    def __init__(self, in_channels, out_channels, bridge_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2 = nn.Conv2d(
            out_channels + bridge_channels, out_channels, 3, 1, 1)

    def forward(self, x, bridge):
        y = self.conv1(x)
        y = self.up(y)
        y = torch.cat((y, bridge), dim=1)
        y = self.conv2(y)
        return y


class SegnetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.head = nn.Conv2d(in_channels[0], out_channels, 3, 1, 1)
        self.body = nn.ModuleList()
        for ic in in_channels:
            self.body.append(FeatureUp(out_channels, out_channels, ic))
        self.tail = nn.Conv2d(out_channels, 1, 3, 1, 1)

    def forward(self, in_features):
        in_features = in_features[::-1]
        skip = [self.head(in_features[0])]
        for net, feat in zip(self.body, in_features[1:]):
            skip.append(net(skip[-1], feat))
        matte = self.tail(skip[-1])
        return matte


def build_segnet(cfg, in_channels):
    return SegnetHead(in_channels, cfg.MODEL.SEGNET.OUT_CHANNELS)


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label 2D tensor to a one-hot 3D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor.

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> tgm.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        labels = labels[:, 0]
    if not labels.dtype == torch.int64:
        labels = labels.long()
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
        [2] https://github.com/zhezh/focalloss/blob/master/focalloss.py
        [3] https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/focal.html
    """

    def __init__(self, alpha: float, gamma: Optional[float] = 2.0,
                 reduction: Optional[str] = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        if input.shape[1] > 1:
            input_soft = torch.softmax(input, dim=1) + self.eps

            # create the labels one hot tensor
            target_one_hot = one_hot(target, num_classes=input.shape[1],
                                     device=input.device, dtype=input.dtype)
        else:
            input_soft = torch.sigmoid(input)
            input_soft = torch.cat((1 - input_soft, input_soft), 1)

            # create the labels one hot tensor
            target_one_hot = one_hot(target, num_classes=2,
                                     device=input.device, dtype=input.dtype)

        # compute the actual focal loss
        weight = torch.pow(1. - input_soft, self.gamma)
        focal = -self.alpha * weight * torch.log(input_soft + self.eps)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        loss = -1
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        if input.shape[1] > 1:
            input_soft = torch.softmax(input, dim=1)

            # create the labels one hot tensor
            target_one_hot = one_hot(target, num_classes=input.shape[1],
                                     device=input.device, dtype=input.dtype)
        else:
            input_soft = torch.sigmoid(input)
            target_one_hot = target

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)


class FocalDiceLoss(nn.Module):
    """Combine focal loss with dice loss.
    """

    def __init__(self, alpha, gamma, dice_weight=0.5):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha, gamma, reduction='mean')
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, input, target):
        fl = self.focal_loss(input, target)
        dl = self.dice_loss(input, target) * self.dice_weight
        return fl + dl


def gather_instance_to_global_mask(pred_mask_logits, instances):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)

    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        boxes = torch.tensor(
            [0, 0, *instances_per_image.image_size]).repeat([len(instances_per_image), 4])
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            boxes, mask_side_len
        ).to(device=pred_mask_logits.device).sum(0).clamp(0, 1)
        gt_masks.append(gt_masks_per_image)
    return torch.stack(gt_masks)


def segnet_loss(pred_mask_logits, instances):
    gt_masks = gather_instance_to_global_mask(pred_mask_logits, instances)

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
    )
    return mask_loss
