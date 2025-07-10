
import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from module import logger 
from compute_mmcm.compute_mmcm import ComputeMMCM


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def main(args):

    """setup"""
    logger.configure(args.output_folder)

    logger.log(f"> Executed command: '{sys.argv}' ")
    logger.log(f"> pred_path: '{args.pred_path}'")

    if args.dataset_name == "h36m":
        stride = 25
    elif args.dataset_name == "amass":
        stride = 120
    compute_mmcm = ComputeMMCM(dataset_name=args.dataset_name, store_folder=logger.get_dir())
    compute_mmcm.clustering(data_config_path=args.data_config_path, stride=stride)

    # load numpy files
    npy_files = os.listdir(args.pred_path)
    npy_files = sorted(npy_files, key=lambda fn: (int(fn.split('_')[0]), 0 if fn.endswith("_obs.npy") else 1))

    for idx in tqdm(range(len(npy_files)//2)):
        obs_file = os.path.join(args.pred_path, npy_files[2*idx])     # example：0_['S9']_['Sitting']_0to124_obs.npy
        pred_file = os.path.join(args.pred_path, npy_files[2*idx+1])   # example：0_['S9']_['Sitting']_0to124_pred.npy

        assert obs_file.split("/")[-1].split("_")[:-1] == pred_file.split("/")[-1].split("_")[:-1], f"obs_file: {obs_file}, pred_file: {pred_file}"

        obs_numpy = np.load(obs_file)
        pred_numpy = np.load(pred_file)
        pred = torch.from_numpy(pred_numpy).squeeze(0)
        obs = torch.from_numpy(obs_numpy).repeat(pred.shape[0], 1, 1, 1, 1)
        compute_mmcm.metric_result_store(obs=obs, pred=pred, sample_idx=idx)


    mmcm, coverage_rate, validity_ratet  = compute_mmcm.return_metric()

    # log
    logger.log(f"MMCM: {mmcm.mean()}")
    logger.log(f"Coverage Rate: {coverage_rate.mean()}")
    logger.log(f"Validity Rate: {validity_ratet.mean()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default="baseline_output/comusion/h36m/npy/")
    parser.add_argument('--output_folder', type=str, default="")
    parser.add_argument('--data_config_path', type=str, default="compute_mmcm/default_parms/h36m/h36_config.json")
    parser.add_argument('--dataset_name', type=str, default="h36m")
    args = parser.parse_args()
    main(args)
