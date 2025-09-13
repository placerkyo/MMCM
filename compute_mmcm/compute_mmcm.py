
import os
import torch
import numpy as np
import hdbscan
import umap.umap_ as umap
from tqdm import tqdm
import pickle
import joblib


# # 2つ上の階層を参照パスに追加
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.torch import *
import autoencoder.models as module_arch
from utils import read_json
from parse_config import ConfigParser
from module import logger 
from data_loader.h36_module import prepare_data
from metrics.evaluation import get_multimodal_gt


class AEModelParams():
    def __init__(self):
        self.model_config_path = None
        self.model_name = None
        self.device = None
        self.autoencoder_model = None
        self.checkpoint = None
        self.frames = None
        self.numpy_train_latent_path = None

        # fix random seeds for reproducibility
        self.DEFAULT_SEED = 6

    def set_params(self, model_config_path, model_name, autoencoder_model, 
                   checkpoint, frames, device, numpy_train_latent_path):
        self.model_config_path = model_config_path
        self.model_name = model_name
        self.device = device
        self.autoencoder_model = autoencoder_model
        self.checkpoint = checkpoint
        self.frames = frames
        self.numpy_train_latent_path = numpy_train_latent_path
        return self

    def set_default_params(self, dataset_name):
        self.model_config_path = f"compute_mmcm/default_parms/{dataset_name}/autoencoder_config.json"
        if dataset_name == "h36m":
            self.model_name = "checkpoint.pth"
            self.numpy_train_latent_path = "compute_mmcm/default_parms/h36m/train_latent_autoencoder_checkpoint.npy"
            self.frames = 103
        elif dataset_name == "amass":
            self.model_name = "checkpoint.pth"
            self.numpy_train_latent_path = "compute_mmcm/default_parms/amass/train_latent_autoencoder_checkpoint.npy"
            self.frames = 123
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder_model, self.checkpoint = self.prepare_checkpoints()
        self.autoencoder_model.load_state_dict(self.checkpoint['state_dict'])
        self.autoencoder_model = self.autoencoder_model.to(self.device)
        self.autoencoder_model.eval()
        return self

    def prepare_checkpoints(self):
        """
        Prepare the model checkpoints.
        """
        config_dict = read_json(self.model_config_path)
        config_dict["seed"] = self.DEFAULT_SEED
        config_dict["config_path"] = self.model_config_path
        config_dict["trainer"]["batch_size"] = 1
        config = ConfigParser(config_dict)
        model = config.init_obj('arch', module_arch)

        checkpoint = torch.load(os.path.join(os.path.dirname(self.model_config_path), self.model_name), weights_only=False)
        logger.log(f"Load model: {self.model_name}")
        return model, checkpoint


class HDBSCANParams():
    def __init__(self):
        self.min_cluster_size = None
        self.min_samples = None

    def set_params(self, min_cluster_size, min_samples):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        return self

    def set_default_params(self, dataset_name):
        if dataset_name == "h36m":
            self.min_cluster_size = 15
            self.min_samples = 1
        elif dataset_name == "amass":
            self.min_cluster_size = 50
            self.min_samples = 1
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}. Please set the parameters manually.")
        return self



class UMAPParams():
    def __init__(self):
        self.use_umap = None
        self.n_neighbors = None
        self.n_components = None
        self.random_state = None
        self.umap_path = None
        self.numpy_train_latent_path = None

    def set_params(self, use_umap, n_neighbors, n_components, random_state, 
                   umap_path, numpy_train_latent_path):
        self.use_umap = use_umap
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.random_state = random_state
        self.umap_path = umap_path
        self.numpy_train_latent_path = numpy_train_latent_path
        return self

    def set_default_params(self, dataset_name):
        self.use_umap = True
        self.n_neighbors = 15
        self.n_components = 2
        self.random_state = 0
        self.umap_path = f"compute_mmcm/default_parms/{dataset_name}/trained_umap_nneighbors=15_ncomponents=2.pkl"
        self.numpy_train_latent_path = f"compute_mmcm/default_parms/{dataset_name}/train_latent_umap_nneighbors=15_ncomponents=2.npy"
        return self

class MMGTParams():
    def __init__(self):
        self.frames = None
        self.threshold = None

    def set_params(self, frames, threshold):
        self.frames = frames
        self.threshold = threshold
        return self
    
    def set_default_params(self, dataset_name):
        if dataset_name == "h36m":
            self.mmgt_frames = 1
            self.mmgt_threshold = 0.5
            self.test_stride = 25
        elif dataset_name == "amass":
            self.mmgt_frames = 1
            self.mmgt_threshold = 0.4
            self.test_stride = 120
        self.path = f"compute_mmcm/default_parms/{dataset_name}/mmgt_labels_frames={self.mmgt_frames}_thresh={self.mmgt_threshold}.csv"
        return self



class ComputeMMCM():
    def __init__(
            self,
            dataset_name: str = "h36m",
            ae_model_parms: AEModelParams=None,
            hdbscan_parms: HDBSCANParams=None,
            umap_parms: UMAPParams=None,
            mmgt_parms: MMGTParams=None,
            store_folder: str = "",
            ):
        self.dataset_name = dataset_name
        # Initialize the class with the model parameters
        self.ae_model_parms = AEModelParams().set_default_params(self.dataset_name) if ae_model_parms is None else ae_model_parms
        self.hdbscan_parms = HDBSCANParams().set_default_params(self.dataset_name) if hdbscan_parms is None else hdbscan_parms
        self.umap_parms = UMAPParams().set_default_params(self.dataset_name) if umap_parms is None else umap_parms
        self.mmgt_parms = MMGTParams().set_default_params(self.dataset_name) if mmgt_parms is None else mmgt_parms

        self.store_folder = store_folder

        logger.log((f"Done setting up the model parameters"))

    def clustering(self, data_config_path, stride=25):
        """
        Fit the HDBSCAN model to the training data.
        """        
        self.data_config_path = data_config_path
        # prepare data loader
        data_config = read_json(data_config_path)
        data_config = ConfigParser(data_config, save=False)
        data_loader_name = f"data_loader_training"
        data_config[data_loader_name]["args"]["stride"] = stride
        data_loader = prepare_data(data_config, data_loader_name, shuffle=False, drop_last=False, num_workers=0, batch_size=1, )
        
        logger.log(f"length of data_loader: {len(data_loader)}")

        # input to the autoencoder
        latent = self.return_latent_autoencoder(data_loader)

        # input to the umap
        if self.umap_parms.use_umap:
            self.umap = self.return_trained_umap(latent)
            # Save the UMAP model
            joblib.dump(self.umap, os.path.join(self.store_folder, "trained_umap.pkl"))

            # Transform the data
            latent = self.return_latent_umap(self.umap, latent)

        # input to the hdbscan
        self.clusterer = self.return_hdbscan()
        self.clusterer.fit(latent)

        # Save the clusterer model
        with open(os.path.join(self.store_folder, "hdbscan_model.pkl"), "wb") as f:
            pickle.dump(self.clusterer, f)

    def return_mmgt_labels(self, frames=1, threshold=0.5):
        """
        Get the ground truth trajectories for the multimodal evaluation.
        """
        path = self.mmgt_parms.path
        # Check if the mmgt labels file already exists
        if os.path.exists(path):
            logger.log(f"Load mmgt labels from {path}")
            with open(path, 'r') as f:
                mmgt_labels = f.readlines()
                mmgt_labels = [line.strip().split(",") for line in mmgt_labels]
            for i in range(len(mmgt_labels)):
                mmgt_labels[i] = [int(label) for label in mmgt_labels[i]]
            return mmgt_labels     # mmgt_labels[i] = [sample_idx, num_mmgt, num_unique_labels, label1, label2, ...]
        

        def label_result(obs, pred):
            """
            Calculate the labels for the multimodal evaluation.
            """
            dist_mat_torch = self.calculate_dist_mat(obs, pred)

            # number of clusters
            nearest_dist_torch, nearest_idx_torch = torch.min(dist_mat_torch, dim=1)
            labels = nearest_idx_torch.detach().cpu().numpy()
            return labels

        data_config = read_json(self.data_config_path)
        data_config = ConfigParser(data_config, save=False)
        data_loader_name = f"data_loader_test"
        data_config[data_loader_name]["args"]["stride"] = self.mmgt_parms.test_stride
        data_loader = prepare_data(data_config, data_loader_name, shuffle=False, drop_last=False, num_workers=0, batch_size=1, )
        batches_toenumerate = enumerate(data_loader)
        len_data_loader = len(data_loader)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        traj_gt_arr = get_multimodal_gt(data_loader, threshold, device=device, split="test", num_frames=self.mmgt_parms.mmgt_frames)

        counter = 0
        for nbatch, batch in tqdm(batches_toenumerate, total=len_data_loader):
            data, target, extra = batch
            mm_traj_list = traj_gt_arr[counter:counter + data.shape[0]]
            
            # Check if there is exactly one mm_traj for each data sample
            assert len(mm_traj_list) == 1, f"mm_traj length is not 1: {len(mm_traj_list)}"

            mm_traj = mm_traj_list[0]
            data = data.repeat(mm_traj.shape[0], 1, 1, 1, 1)

            labels = label_result(obs=data, pred=mm_traj)

            # Write the results to the file
            with open(path, 'a') as f:
                f.write(f"{counter},{mm_traj.shape[0]},{len(np.unique(labels))},{','.join(map(str, labels))}\n")
            counter += target.shape[0]
        logger.log(f"Load mmgt labels from {path}")

        with open(path, 'r') as f:
            mmgt_labels = f.readlines()
            mmgt_labels = [line.strip().split(",") for line in mmgt_labels]
        for i in range(len(mmgt_labels)):
            mmgt_labels[i] = [int(label) for label in mmgt_labels[i]]
        return mmgt_labels     # mmgt_labels[i] = [sample_idx, num_mmgt, num_unique_labels, label1, label2, ...]


    def return_hdbscan(self):
        """
        Get the HDBSCAN model.
        """
        logger.log(f"set HDBSCAN parameters: min_cluster_size={self.hdbscan_parms.min_cluster_size}, min_samples={self.hdbscan_parms.min_samples}")
        return hdbscan.HDBSCAN(min_cluster_size=self.hdbscan_parms.min_cluster_size, min_samples=self.hdbscan_parms.min_samples, prediction_data=True)
    
    def return_umap(self):
        """
        Get the UMAP model.
        """
        logger.log(f"set UMAP parameters: n_neighbors={self.umap_parms.n_neighbors}, n_components={self.umap_parms.n_components}")
        return umap.UMAP(n_neighbors=self.umap_parms.n_neighbors, n_components=self.umap_parms.n_components, random_state=0, transform_queue_size=1.0, transform_seed=0)

    def return_trained_umap(self, latent_all):
        """
        Get the trained UMAP model.
        """

        #　Check if the UMAP model already exists
        if os.path.exists(self.umap_parms.umap_path):
            logger.log(f"Load UMAP model from {self.umap_parms.umap_path}")
            umap = joblib.load(self.umap_parms.umap_path)
            return umap

        # If the UMAP model does not exist, create a new one
        umap = self.return_umap()
        umap.fit(latent_all)

        # Save the UMAP model
        os.makedirs(os.path.dirname(self.umap_parms.umap_path), exist_ok=True)
        joblib.dump(umap, self.umap_parms.umap_path)

        return umap

    def return_latent_umap(self, umap, latent_all):
        """
        Get the latent representation of the data using the UMAP model.
        """

        #　Check if the latent file already exists
        if os.path.exists(self.umap_parms.numpy_train_latent_path):
            logger.log(f"Load latent umap from {self.umap_parms.numpy_train_latent_path}")
            latent_all = np.load(self.umap_parms.numpy_train_latent_path)
            return latent_all

        # Transform the data
        latent_all_embed = []
        for x in latent_all:
            latent_all_embed.append(umap.transform(x.reshape(1, -1)))
        latent_all = np.vstack(latent_all_embed)

        # Save the latent_all to numpy file
        os.makedirs(os.path.dirname(self.umap_parms.numpy_train_latent_path), exist_ok=True)
        np.save(self.umap_parms.numpy_train_latent_path, latent_all)

        return latent_all
    
    def return_latent_autoencoder(self, data_loader):
        """
        Get the latent representation of the data using the autoencoder model.
        """

        #　Check if the latent file already exists
        if os.path.exists(self.ae_model_parms.numpy_train_latent_path):
            logger.log(f"Load latent autoencoder from {self.ae_model_parms.numpy_train_latent_path}")
            latent_all = np.load(self.ae_model_parms.numpy_train_latent_path)
            return latent_all

        latent_all = []
        batches_toenumerate = enumerate(data_loader)
        len_data_loader = len(data_loader)
        
        for nbatch, batch in tqdm(batches_toenumerate, total=len_data_loader):
            data, target, extra = batch
            
            assert self.ae_model_parms.frames >= target.shape[1], f"frames ({self.ae_model_parms.frames}) must be greater than or equal to target.shape[1] ({target.shape[1]})"
            if self.ae_model_parms.frames > target.shape[1]:
                target = torch.cat([data[:,-(self.ae_model_parms.frames-target.shape[1]):,...], target], dim=1)

            data, target = data.to(self.ae_model_parms.device), target.to(self.ae_model_parms.device)
  
            latent  = self.ae_model_parms.autoencoder_model.encode(target, deterministic=True)
            latent_all.append(latent.cpu().detach().numpy())

        latent_all = np.concatenate(latent_all, axis=0)

        # Save the latent_all to numpy file
        os.makedirs(os.path.dirname(self.ae_model_parms.numpy_train_latent_path), exist_ok=True)
        np.save(self.ae_model_parms.numpy_train_latent_path, latent_all)

        return latent_all

    def calculate_dist_mat(self, obs, pred):
        """
        Calculate the clustering results.
        """

        def _check_shape(x, name):
            assert x.ndim == 5,           \
                f"{name} must be 5-D [B,T,1,16,3]; got ndim={x.ndim}"
            assert x.shape[2:] == (1, 16, 3) or x.shape[2:] == (1, 21, 3), \
                f"{name} last three dims must be (1,16,3); got {x.shape[2:]}"
        _check_shape(obs,  "obs")
        _check_shape(pred, "pred")

        target = torch.cat([obs[:,-(self.ae_model_parms.frames-pred.shape[1]):,...], pred], dim=1)
        target = target.to(self.ae_model_parms.device)       # [b, t, 1, 16, 3]

        # Autoencoder
        with torch.no_grad():
            latent = self.ae_model_parms.autoencoder_model.encode(target, deterministic=True)   # [b, t, 1, 16, 3] -> [b, dim_au]

        # UMAP
        if self.umap_parms.use_umap:
            latent_np = latent.cpu().detach().numpy()
            latent_umap_np = self.umap.transform(latent_np)
            latent = torch.from_numpy(latent_umap_np).to(self.ae_model_parms.device)

        # calculate the distance matrix
        exemplars_list = self.clusterer.exemplars_
        centers_np = np.vstack([ex.mean(axis=0) for ex in exemplars_list])
        centers = torch.from_numpy(centers_np).to(self.ae_model_parms.device)
        centers = centers.to(dtype=latent.dtype)
        dist_mat = torch.cdist(latent, centers, p=2.0)

        return dist_mat
    
    def metric_result_store(self, obs, pred, sample_idx, reset=False):
        """
        Store the metrics.
        """

        # If reset flag is set or metrics attribute doesn't exist, initialize all result containers
        if reset or not hasattr(self, 'metrics'):
            self.metrics = np.array([])
            self.nearest_dist = []
            self.labels = []
            self.coverage_rate = np.array([])
            self.validity_rate = np.array([])
            self.covervalidration_mmgt = np.array([])
            self.mmcm = np.array([])

        # Ensure mmgt labels are computed and stored
        if not hasattr(self, 'mmgt_labels'):
            self.mmgt_labels = self.return_mmgt_labels(frames=self.mmgt_parms.mmgt_frames, threshold=self.mmgt_parms.mmgt_threshold)

        # Compute distance matrix between observations and predictions
        dist_mat_torch = self.calculate_dist_mat(obs, pred)
        device = dist_mat_torch.device


        # Find nearest cluster for each sample
        nearest_dist_torch, nearest_idx_torch = torch.min(dist_mat_torch, dim=1)
        B, K = dist_mat_torch.shape
        labels_torch = torch.full((B,), -1, dtype=torch.int64, device=device)

        # Apply dataset-specific distance threshold to assign cluster labels
        if self.dataset_name == "h36m":
            mask = nearest_dist_torch <= 1.024
        elif self.dataset_name == "amass":
            mask = nearest_dist_torch <= 3.14
        labels_torch[mask] = nearest_idx_torch[mask]

        # Identify which samples got a valid cluster assignment
        positive_mask = labels_torch != -1
        if positive_mask.any():
            # Extract unique clusters among positively assigned samples
            unique_clusters_torch = torch.unique(labels_torch[positive_mask])
        else:
            unique_clusters_torch = torch.tensor([], dtype=torch.int64, device=device)

        # Store nearest distances and labels as numpy arrays
        self.nearest_dist.append(nearest_dist_torch.detach().cpu().numpy())
        self.labels.append(labels_torch.detach().cpu().numpy())

        # Retrieve ground-truth cluster labels for this sample
        mmgt_label_list = self.mmgt_labels[sample_idx][3:]  # List[int]
        mmgt_label_torch = torch.tensor(mmgt_label_list, dtype=torch.int64, device=device)  # [M]
        mmgt_unique_clusters = torch.unique(mmgt_label_torch)  # [M_unique]

        # Count how many assigned labels correspond to any ground-truth cluster
        isin_mask = torch.isin(labels_torch, mmgt_unique_clusters)  # [B], bool
        num_clusters_inmmgt = int(isin_mask.sum().detach().cpu().item())

        # Compute the number of ground-truth clusters covered by assigned labels
        if unique_clusters_torch.numel() > 0:
            intersection = torch.tensor(
                [u.item() for u in unique_clusters_torch if (u.unsqueeze(0) == mmgt_unique_clusters.unsqueeze(1)).any()],
                device=device,
                dtype=torch.int64
            )
            num_mmgt_clusters_inlabels = int(intersection.numel())
        else:
            num_mmgt_clusters_inlabels = 0

        # Calculate coverage rate: fraction of GT clusters found in predictions
        if mmgt_unique_clusters.numel() > 0:
            coverration_val = num_mmgt_clusters_inlabels / float(mmgt_unique_clusters.numel())
        else:
            coverration_val = 0.0

        # Calculate validity rate: fraction of predictions falling into GT clusters
        validityration_val = num_clusters_inmmgt / float(labels_torch.shape[0])

        # Compute F1-like multimodality metric (MMCM)
        mmcm = 2 * (coverration_val * validityration_val) / (coverration_val + validityration_val + 1e-8)

        # Append computed rates to stored arrays
        self.coverage_rate = np.concatenate(
            (self.coverage_rate, np.array([coverration_val], dtype=np.float32)), axis=0
        )
        self.validity_rate = np.concatenate(
            (self.validity_rate, np.array([validityration_val], dtype=np.float32)), axis=0
        )
        self.mmcm = np.concatenate(
            (self.mmcm, np.array([mmcm], dtype=np.float32)), axis=0
        )

    def return_metric(self):
        """
        Return the metrics.
        """
        coverage_rate = np.mean(self.coverage_rate)
        validity_ratet = np.mean(self.validity_rate)
        mmcm = np.mean(self.mmcm)
        return mmcm, coverage_rate, validity_ratet

    def return_metric_permotion(self):
        """
        Return all the results of the clustering.
        """
        return {
            "coverage_rate": self.coverage_rate,
            "validity_rate": self.validity_rate,
            "mmcm": self.mmcm,
        }
    
    def return_labels(self):
        """
        Return the labels and distance of the clustering.
        """
        return self.labels, self.nearest_dist

    def reset_results(self):
        """
        Reset the results of the clustering.
        """
        self.metrics = np.array([])
        self.nearest_dist = []
        self.labels = []
        self.coverage_rate = np.array([])
        self.validity_rate = np.array([])
        self.covervalidration_mmgt = np.array([])
        self.mmcm = np.array([])

