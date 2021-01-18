import numpy as np
import pycocotools.mask as mask_util
import torch
import logging

logger = logging.getLogger(f'detectron2.{__name__}')


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


def progress(x, total, freq=0.1):
    if x % int(total * freq + 0.5) == 0 and x > 0:
        logger.info(f"progress: {x}/{total}")
