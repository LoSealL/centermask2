import logging
import os
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode
from PIL import Image

from .utils import progress, seg_to_box

logger = logging.getLogger(f'detectron2.{__name__}')


def load_cihp_dataset(image_path: Path, human_id_path: Path):
    datadict = []
    files = sorted(image_path.glob('*.jpg'))
    logger.info(f"loading CIHP dataset, glob {len(files)} images...")
    for prog, img in enumerate(files):
        progress(prog, len(files))
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
    register_cihp_instance('cihp_train', _root / 'CIHP' / 'Training')
    register_cihp_instance('cihp_val', _root / 'CIHP' / 'Validation')
