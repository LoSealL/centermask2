# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Youngwan Lee (ETRI), 2020. All Rights Reserved.
from pathlib import Path

import tqdm
from detectron2.engine import default_argument_parser, default_setup
from PIL import Image
from torchvision.io.image import write_jpeg

from centermask.config import get_cfg
from centermask.data import build_x_train_loader


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    print("DATASETS: ", cfg.DATASETS.TRAIN)
    loader = build_x_train_loader(cfg)
    for i, items in tqdm.tqdm(enumerate(loader)):
        if i % 128 != 0:
            continue
        save_dir = Path(cfg.OUTPUT_DIR) / f"{items[0]['image_id']}"
        save_dir.mkdir(exist_ok=True, parents=True)
        write_jpeg(items[0]['image'], str(save_dir / f'img{i}.jpg'))
        for j, mask in enumerate(items[0]['instances'].gt_masks):
            Image.fromarray(mask.cpu().numpy()).save(save_dir / f'mask{j}.png')


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
