
import argparse
import os
import sys
import torch
import numpy as np
import hdbscan
from hdbscan.prediction import approximate_predict
import umap.umap_ as umap
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from copy import deepcopy
import pickle
import joblib
import time


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.torch import *
import autoencoder.models as module_arch
from utils import read_json, set_global_seed, update_config_with_arguments
from parse_config import ConfigParser
from module import logger 
from data_loader.h36_module import prepare_data

from module import logger


# fix random seeds for reproducibility
DEFAULT_SEED = 6

# this will be overriden in config file only when set as arguments
ARGS_CONFIGPATH = dict( # alias for CLI argument: route in config file
    name=("name", ),
    batch_size=("trainer", "batch_size"),
    epochs=("trainer", "epochs"),
)
ARGS_TYPES = dict(
    name=str,
    batch_size=int,
    epochs=int,
)


PARAM_GRID = {
    "umap_dim": [2, 4, 8, 16],
    "min_cluster_size": [5, 10, 15, 20, 25, 30, 40, 50, 60],
    "min_samples":      [1, 2, 3, 4, 5, 7, 10, 15],
}


class Clusterer:
    def __init__(
        self, 
        data_config, 
        args,
        store_folder=None,):

        self.data_config = data_config
        self.args = args
        self.store_folder = store_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prepare model
        self.autoencoder_model, self.checkpoint = self.prepare_checkpoints()
        self.autoencoder_model.load_state_dict(self.checkpoint['state_dict'])
        self.autoencoder_model = self.autoencoder_model.to(self.device)
        self.autoencoder_model.eval()

        self.umap = None
        self.hdbscan = None
                    
        # prepare data loader
        self.data_loader_name = f"data_loader_training"
        self.data_config[self.data_loader_name]["args"]["stride"] = self.args.stride
        self.data_loader = prepare_data(self.data_config, self.data_loader_name, shuffle=False, drop_last=False, num_workers=0, batch_size=self.args.batch_size, )
        data_loader_name = f"data_loader_test"
        self.testdata_loader = prepare_data(self.data_config, data_loader_name, shuffle=False, drop_last=False, num_workers=0, batch_size=self.args.batch_size, )
        logger.log(f"length of data_loader: {len(self.data_loader)}, length of testdata_loader: {len(self.testdata_loader)}")

        self.best_setting = {"umap_dim": None, "min_cluster_size": None, "min_samples": None, "sum_stab": -1}


    def parmsearch(self):
        """
        Perform a parameter search for HDBSCAN.
        """
        # Create a list to store the results
        results = []

        # Iterate over all combinations of parameters
        for n_component in PARAM_GRID["umap_dim"]:
            for min_cluster_size in PARAM_GRID["min_cluster_size"]:
                for min_samples in PARAM_GRID["min_samples"]:
                    self.n_components = n_component
                    self.min_cluster_size = min_cluster_size
                    self.min_samples = min_samples

                    # Fit the model and get the results
                    result = self.fit_hdbscan()
                    test_noise_ratio, test_num_clusters = self.clustering()
                    # 2重になってるから意味ない
                    result["test_noise"] = test_noise_ratio
                    result["test_num_clusters"] = test_num_clusters

                    results.append(result)

                    logger.log(f"umap_dim: {n_component}, min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, mean_stab: {result['mean_stab']}, sum_stab: {result['sum_stab']}, noise: {result['noise']}, #clust: {result['#clust']}, test_noise: {test_noise_ratio}, test_num_clusters: {test_num_clusters}")

                    if result["sum_stab"] > self.best_setting["sum_stab"] and test_noise_ratio <= 0.33:
                        self.best_setting = {
                            "umap_dim": n_component,
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                            "sum_stab": result["sum_stab"],
                        }
                        logger.log(f"New best setting found: {self.best_setting}")

        # Save the results to a CSV file
        with open(os.path.join(self.store_folder, "hdbscan_results.csv"), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["min_cluster_size", "min_samples", "mean_stab", "sum_stab", "noise", "#clust", "test_noise", "test_num_clusters"])
            for result in results:
                writer.writerow([result["min_cluster_size"], result["min_samples"], result["mean_stab"], result["sum_stab"], result["noise"], result["#clust"], result["test_noise"], result["test_num_clusters"]])

        logger.log("#"*20)
        logger.log(f"Best setting: {self.best_setting}")
        logger.log("#"*20)

    def return_hdbscan(self):
        logger.log(f"set HDBSCAN parameters: min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}")
        return hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, prediction_data=True)
    
    def return_umap(self):
        logger.log(f"set UMAP parameters: n_neighbors={15}, n_components={self.n_components}")
        return umap.UMAP(n_neighbors=15, n_components=self.n_components, random_state=0, transform_queue_size=1.0, transform_seed=0)

    def return_trained_umap(self, latent_all):
        file_name = f"trained_umap_nneighbors={15}_ncomponents={self.n_components}.pkl"
        umap_path = os.path.join(os.path.dirname(self.args.model_config_path), "trained_umap",file_name)

        #　Check if the UMAP model already exists
        if os.path.exists(umap_path):
            logger.log(f"Load UMAP model from {umap_path}")
            umap = joblib.load(umap_path)
            return umap

        # If the UMAP model does not exist, create a new one
        umap = self.return_umap()
        umap.fit(latent_all)

        # Save the UMAP model
        os.makedirs(os.path.dirname(umap_path), exist_ok=True)
        joblib.dump(umap, umap_path)

        return umap

    def return_latent_umap(self, umap, latent_all, split):   # split = "train" or "test"
        file_name = f"{split}_latent_umap_nneighbors={15}_ncomponents={self.n_components}.npy"
        numpy_train_latent_path = os.path.join(os.path.dirname(self.args.model_config_path), "latent_umap", file_name)

        #　Check if the latent file already exists
        if os.path.exists(numpy_train_latent_path):
            logger.log(f"Load latent umap from {numpy_train_latent_path}")
            latent_all = np.load(numpy_train_latent_path)
            return latent_all

        # Transform the data
        latent_all_embed = []
        for x in latent_all:
            latent_all_embed.append(umap.transform(x.reshape(1, -1)))
        latent_all = np.vstack(latent_all_embed)

        # Save the latent_all to numpy file
        os.makedirs(os.path.dirname(numpy_train_latent_path), exist_ok=True)
        np.save(numpy_train_latent_path, latent_all)

        return latent_all
    
    def return_latent_autoencoder(self, split):   # split = "train" or "test"

        file_name = f"{split}_latent_autoencoder_{self.args.model_name.split('.')[0]}_stride={self.args.stride}.npy"
        numpy_train_latent_path = os.path.join(os.path.dirname(self.args.model_config_path), "latent_autoencoder",file_name)

        #　Check if the latent file already exists
        if os.path.exists(numpy_train_latent_path):
            logger.log(f"Load latent autoencoder from {numpy_train_latent_path}")
            latent_all = np.load(numpy_train_latent_path)
            return latent_all

        latent_all = []
        # for each batch
        if split == "train":
            batches_toenumerate = enumerate(self.data_loader)
            len_data_loader = len(self.data_loader)
        else:
            batches_toenumerate = enumerate(self.testdata_loader)
            len_data_loader = len(self.testdata_loader)
        for nbatch, batch in tqdm(batches_toenumerate, total=len_data_loader):
            data, target, extra = batch
            
            assert self.args.frames >= target.shape[1], f"frames ({self.args.frames}) must be greater than or equal to target.shape[1] ({target.shape[1]})"
            if self.args.frames > target.shape[1]:
                target = torch.cat([data[:,-(self.args.frames-target.shape[1]):,...], target], dim=1)
            target = target.to(self.device)
  
            latent  = self.autoencoder_model.encode(target, deterministic=True)
            latent_all.append(latent.cpu().detach().numpy())

        latent_all = np.concatenate(latent_all, axis=0)

        # Save the latent_all to numpy file
        os.makedirs(os.path.dirname(numpy_train_latent_path), exist_ok=True)
        np.save(numpy_train_latent_path, latent_all)

        return latent_all


    def fit_hdbscan(self):
        """
        Fit the HDBSCAN model to the training data.
        """
        latent_all = self.return_latent_autoencoder("train")

        self.umap = self.return_trained_umap(latent_all)

        # Transform the data
        latent_all = self.return_latent_umap(self.umap, latent_all, "train")

        self.clusterer = self.return_hdbscan()
        self.clusterer.fit(latent_all)
        labels = self.clusterer.labels_

        noise_ratio = np.mean(labels == -1)
        mean_stability = self.clusterer.cluster_persistence_.mean()
        sum_stability  = self.clusterer.cluster_persistence_.sum()

        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "mean_stab": mean_stability,
            "sum_stab": sum_stability,
            "noise": noise_ratio,
            "#clust": len(set(labels))-1,
        }



    def clustering(self, hdbscan_model_path=None, umap_model_path=None):
        """
        Load the trained HDBSCAN and UMAP models, and perform clustering on the test data.
        """

        if hdbscan_model_path is not None:
            with open(hdbscan_model_path, "rb") as f:
                self.clusterer = pickle.load(f)
        if umap_model_path is not None:
            self.umap = joblib.load(umap_model_path)

        assert self.clusterer is not None, "HDBSCAN model must be loaded or trained before clustering."
        assert self.umap is not None, "UMAP model must be loaded or trained before clustering."

        latent_all = self.return_latent_autoencoder("test")

        # Transform the data
        latent_all = self.return_latent_umap(self.umap, latent_all, "test")

        # HDBSCAN
        all_labels, all_strengths = approximate_predict(self.clusterer, latent_all)

        noise_ratio = np.mean(all_labels == -1)
        num_clusters = len(set(all_labels)) - 1

        return noise_ratio, num_clusters


    def prepare_checkpoints(self):
        """
        Prepare the model checkpoints.
        """
        config_dict = read_json(self.args.model_config_path)
        update_config_with_arguments(config_dict, self.args, ARGS_TYPES, ARGS_CONFIGPATH)
        config_dict["seed"] = DEFAULT_SEED
        config_dict["config_path"] = self.args.model_config_path
        config = ConfigParser(config_dict)
        model = config.init_obj('arch', module_arch)

        checkpoint = torch.load(os.path.join(os.path.dirname(self.args.model_config_path), self.args.model_name))
        logger.log(f"Load model: {self.args.model_name}")
        return model, checkpoint



if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-o', '--output_folder', type=str, default="")
    parser.add_argument('--data_config_path', type=str, default="compute_mmcm/default_parms/h36m/h36_config.json")
    parser.add_argument('-d', '--data', default='test')
    parser.add_argument('--dataset_split', type=str, default='training')
    parser.add_argument('--stride', type=int, default=25)
    # autoencoderのパラメータ
    parser.add_argument('--model_config_path', type=str, default="compute_mmcm/default_parms/h36m/autoencoder_config.json")
    parser.add_argument('--model_name', type=str, default="checkpoint-epoch3000.pth")
    parser.add_argument('--frames', type=int, default=103)

    args = parser.parse_args()
    assert args.batch_size == 1, f"batch size must be 1 for motionmap"


    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_global_seed(args.seed)

    logger.configure(args.output_folder)
    

    data_config = read_json(args.data_config_path)
    data_config = ConfigParser(data_config, save=False)

    logger.log(f"> Executed command: '{sys.argv}' ")
    logger.log(f"> GPU name: '{torch.cuda.get_device_name()}'")
    
    clusterer = Clusterer(data_config, 
                        args,
                        store_folder=logger.get_dir(),)
    
    clusterer.parmsearch()

    logger.log(f"Total time: {time.time() - start_time} seconds")


