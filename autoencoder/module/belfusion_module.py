

import matplotlib.pyplot as plt
import torch
from utils.torch import *
from utils.util import AverageMeter

import baselines.BeLFusion.models.diffusion as module_diffusion
import data_loader as module_data
import baselines.BeLFusion.models as module_arch
import os
import pandas as pd
import numpy as np
from utils.visualization.generic import AnimationRenderer
from metrics.evaluation import lat_apd, get_multimodal_gt, cmd
import json
from metrics.fid import fid
from tqdm import tqdm
from utils import read_json
from parse_config import ConfigParser

SAMPLERS = {
    "ddpm": "p_sample_loop_progressive",
    "ddim": "ddim_sample_loop_progressive"
}


def get_prediction(obs, pred, model, diffusion, sample_num, pred_length, steps=None, sampler_name="ddpm", silent=False):
    """
    If idces and to_store_folder != None => prediction will be loaded/stored to avoid generating it again.
    """

    # right now we predict 'sample_num' times with our deterministic model
    bs, obs_length, p, j, f = obs.shape
    diffusion_steps = diffusion.num_timesteps
    num_steps = diffusion_steps if steps is None else len(steps) # if unspecified, all denoising steps are stored

    ys = torch.zeros((bs, sample_num, num_steps, pred_length, p, j, f), device=obs.device)
    all_enc = torch.zeros((bs, sample_num, num_steps, 128), device=obs.device)
    model_args = {
        "obs": obs # for conditioning generation
    }

    toenumerate = range(sample_num) if silent else tqdm(range(sample_num))
    for i in toenumerate:
        shape = (bs, pred_length, p, j, f) # shape -> [N, Seq_length, Partic, Joints, Feat]
        
        sampler = getattr(diffusion, SAMPLERS[sampler_name])

        step_counter = 0
        for s, out in enumerate(sampler(model, shape, progress=False, model_kwargs=model_args, pred=pred)):
            if steps is None or s+1 in steps:
                ys[:, i, step_counter, :] = out["pred_xstart"]
                all_enc[:, i, step_counter] = out["pred_xstart_enc"]
                step_counter += 1
            

    return ys, all_enc

def prepare_model(config, silent=False):

    for i in range(torch.cuda.device_count()):
        if not silent:
            print(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.set_grad_enabled(False)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    diffusion = config.init_obj('diffusion', module_diffusion)

    if not silent:
        print('Loading checkpoint: {} ...'.format(config.resume))
    if ".pth" not in config.resume: # support for models stored in ".p" format
        if not silent:
            print("Loading from a '.p' checkpoint. Only evaluation is supported. Only model weights will be loaded.")
        import pickle
        state_dict = pickle.load(open(config.resume, "rb"))['model_dict']
    else: # ".pth" format
        checkpoint = torch.load(config.resume, map_location=device)
        state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    
    return model, diffusion, device

