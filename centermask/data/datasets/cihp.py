import contextlib
import datetime
import io
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.data import DatasetCatalog
from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.utils.file_io import PathManager
from fvcore.common.file_io import file_lock
from fvcore.common.timer import Timer
from PIL import Image

logger = logging.getLogger(__name__)


def seg_to_box(ins):
    if isinstance(ins, torch.Tensor):
        assert ins.ndim == 2
        y, x = torch.nonzero(ins, as_tuple=True)
    else:
        assert np.ndim(ins) == 2
        y, x = ins.nonzero()

    if not len(x) or not len(y):
        return 0, 0, 0, 0
    return x.min(), y.min(), x.max(), y.max()


def load_cihp_dataset(image_path: Path, human_id_path: Path):
    datadict = []
    for img in image_path.glob('*.jpg'):
        record = {}
        img_head = Image.open(img)
        record['file_name'] = str(img.resolve())
        record['width'] = img_head.width
        record['height'] = img_head.height
        record['image_id'] = int(img.stem)
        del img_head
        # CIHP anno is the same name as the image
        human_ids = list(human_id_path.glob(img.stem + '.png'))
        if len(human_ids) == 0:
            logger.warning(f"No annotation found for {img.stem}, skip it.")
            continue
        assert len(human_ids) == 1, f"Duplicated annotations! {img.stem}"
        mask = np.array(Image.open(human_ids[0]))
        annos = []
        n_humans = mask.max()
        for i in range(n_humans):
            obj = {
                'is_crowd': 0,
                'category_id': 0,  # human id in coco
                'bbox_mode': BoxMode.XYXY_ABS
            }
            human = (mask == i + 1).astype('uint8')
            if np.ndim(human) == 3:
                human = np.squeeze(human, -1)
            box = seg_to_box(human)
            obj['bbox'] = box
            # human = human[box[1]:box[3], box[0]:box[2]]
            human = np.asfortranarray(human)
            rle = mask_util.encode(human)
            obj['segmentation'] = rle
            annos.append(obj)
        record['annotations'] = annos
        datadict.append(record)
    return datadict



def register_cihp_instance(name, cihp_root):
    image_root = Path(cihp_root) / 'Images'
    human_id_root = Path(cihp_root) / 'Human_ids'
    DatasetCatalog.register(
        name, lambda: load_cihp_dataset(image_root, human_id_root))


if __name__.endswith('.cihp'):
    _root = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_cihp_instance('cihp', _root / 'cihp')
