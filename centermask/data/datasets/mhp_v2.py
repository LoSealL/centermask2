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


def load_mhpv2_dataset(image_path: Path, human_id_path: Path):
    datadict = []
    files = sorted(image_path.glob('*.jpg'))
    logger.info(f"loading MHP V2 dataset, glob {len(files)} images...")
    for prog, img in enumerate(files):
        progress(prog, len(files))
        record = {}
        try:
            img_head = Image.open(img)
            record['file_name'] = str(img.resolve())
            record['width'] = img_head.width
            record['height'] = img_head.height
            record['image_id'] = int(img.stem)
            del img_head
        except:
            logger.error(img)
            raise
        # mhpv2 anno is the startswith name as the image
        human_ids = list(human_id_path.glob(img.stem + '_??_??.png'))
        if len(human_ids) == 0:
            logger.warning(f"No annotation found for {img.stem}, skip it.")
            continue
        annos = []
        n_humans = len(human_ids)
        for i in range(n_humans):
            _n_humans, _human_id = human_ids[i].stem.split('_')[-2:]
            assert int(_n_humans) == n_humans, f"mis-matched human number: expected {_n_humans}, found {n_humans}."
            assert int(_human_id) == i + 1, f"mis-matched human id: expected {_human_id}, got {i + 1}."
            mask = np.array(Image.open(human_ids[i]).convert('L'))
            obj = {
                'is_crowd': 0,
                'category_id': 0,  # human id in coco
                'bbox_mode': BoxMode.XYXY_ABS
            }
            human = (mask != 0).astype('uint8')
            box = seg_to_box(human)
            obj['bbox'] = box
            human = np.asfortranarray(human)
            rle = mask_util.encode(human)
            obj['segmentation'] = rle
            annos.append(obj)
        record['annotations'] = annos
        datadict.append(record)
    return datadict


def register_mhpv2_instance(name, root):
    image_root = Path(root) / 'images'
    human_id_root = Path(root) / 'parsing_annos'
    DatasetCatalog.register(
        name, lambda: load_mhpv2_dataset(image_root, human_id_root))


if __name__.endswith('.mhp_v2'):
    _root = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_mhpv2_instance('mhp_v2_train', _root / 'LV-MHP-v2' / 'train')
    register_mhpv2_instance('mhp_v2_val', _root / 'LV-MHP-v2' / 'val')
