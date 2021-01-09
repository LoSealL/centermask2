from typing import List

import torch
import torch.distributed as dist
from detectron2.structures import Instances
from detectron2.utils.comm import get_world_size


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def filter_instance_classes(instances: List[Instances],
                            inc_classes: List[int],
                            is_train=True) -> List[Instances]:
    """Filter instances with the given class ids

    Args:
        instances (List[Instances]): a list of instances
        inc_classes (list[int]): a list of class ids to remain

    Returns:
        List[Instances]: a list of intances removed unwanted classes
    """

    assert isinstance(instances, list), f"instances must be a list"
    assert isinstance(inc_classes, list), f"classes must be a list"

    if not instances or not inc_classes:
        return instances

    if is_train:
        filtered_ins = []
        for ins_per_img in instances:
            classes = ins_per_img.gt_classes
            idx = classes == inc_classes[0]
            for i in inc_classes[1:]:
                idx |= classes == i
            idx = torch.nonzero(idx).flatten()
            filtered_ins.append(ins_per_img[idx])
        return filtered_ins

    return instances


def find_foreground_person(instances: List[Instances]):
    # keep the biggest person
    big_person = []
    for instance_per_image in instances:
        person_idx = torch.nonzero(instance_per_image.pred_classes == 0)
        person_idx = person_idx.flatten()
        fields = instance_per_image.get_fields()
        for k, v in fields.items():
            fields[k] = v[person_idx]
        boxes = instance_per_image.pred_boxes
        area = boxes.area()
        sort_idx = sorted(range(len(boxes)), key=lambda i: area[i], reverse=True)
        if not sort_idx:
            continue
        big_person.append(instance_per_image[sort_idx[0]])
    return big_person
