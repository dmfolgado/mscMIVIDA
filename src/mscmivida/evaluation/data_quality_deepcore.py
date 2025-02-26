import glob

import torch
from loguru import logger

CKPT_FOLDER = "/home/duarte.folgado/dev/ACHILLES/DeepCore/result/"
for f in glob.glob(CKPT_FOLDER + "**/*.ckpt", recursive=True):
    logger.info(f)

checkpoint = torch.load(f, weights_only=False)

checkpoint["subset"]["indices"]  # the dataset indices selected by coreset selection method.
