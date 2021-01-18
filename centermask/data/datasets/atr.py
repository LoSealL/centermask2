import logging
import os
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode
from PIL import Image
from scipy.ndimage import median_filter

from .utils import progress, seg_to_box

logger = logging.getLogger(f'detectron2.{__name__}')


def load_atr_dataset(image_path: Path, anno_path: Path):
    datadict = []
    files = sorted(image_path.glob('*.jpg'))
    logger.info(f"loading ATR dataset, glob {len(files)} images...")
    for prog, img in enumerate(files):
        progress(prog, len(files))
        record = {}
        try:
            img_head = Image.open(img)
            record['file_name'] = str(img.resolve())
            record['width'] = img_head.width
            record['height'] = img_head.height
            record['image_id'] = img.stem
            del img_head
        except:
            logger.error(img)
            raise
        # CIHP anno is the same name as the image
        human_ids = list(anno_path.glob(img.stem + '.png'))
        if len(human_ids) == 0:
            logger.warning(f"No annotation found for {img.stem}, skip it.")
            continue
        assert len(human_ids) == 1, f"Duplicated annotations! {img.stem}"
        mask = np.array(Image.open(human_ids[0]).convert('L'))
        human = (mask != 0).astype('uint8')
        box = seg_to_box(human)
        # human = median_filter(human, size=3)
        human = np.asfortranarray(human)
        rle = mask_util.encode(human)
        obj = {
            'is_crowd': 0,
            'category_id': 0,  # human id in coco
            'bbox_mode': BoxMode.XYXY_ABS,
            'bbox': box,
            'segmentation': rle
        }
        record['annotations'] = [obj]
        datadict.append(record)
    return datadict


def register_atr_instance(name, atr_root):
    image_root = Path(atr_root) / 'JPEGImages'
    human_id_root = Path(atr_root) / 'SegmentationClassAug'
    DatasetCatalog.register(
        name, lambda: load_atr_dataset(image_root, human_id_root))


if __name__.endswith('.atr'):
    _root = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_atr_instance('atr', _root / 'ATR')
