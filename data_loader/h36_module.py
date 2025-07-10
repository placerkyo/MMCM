
import torch
from utils.torch import *

import data_loader as module_data

def prepare_data(config, data_loader_name, shuffle=False, augmentation=0, da_mirroring=0, da_rotations=0, drop_last=True, num_workers=None, batch_size=None, silent=False):

    for i in range(torch.cuda.device_count()):
        if not silent:
            print(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")

    config[data_loader_name]["args"]["shuffle"] = shuffle

    config[data_loader_name]["args"]["da_mirroring"] = da_mirroring
    config[data_loader_name]["args"]["da_rotations"] = da_rotations
    config[data_loader_name]["args"]["augmentation"] = augmentation
    config[data_loader_name]["args"]["drop_last"] = drop_last
    if batch_size is not None:
        config[data_loader_name]["args"]["batch_size"] = batch_size
    if num_workers is not None:
        config[data_loader_name]["args"]["num_workers"] = 0
    data_loader = config.init_obj(data_loader_name, module_data)

    return data_loader
