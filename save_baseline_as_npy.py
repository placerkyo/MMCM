
import os
import numpy as np
from tqdm import tqdm
from data_loader.h36_module import prepare_data
from utils import read_json
from parse_config import ConfigParser

# prepare your model. You should have a function like this in your codebase.
def create_model():
    # Dummy model and device for illustration purposes
    class DummyModel:
        def __call__(self, obs):
            # Dummy prediction: just return input for illustration
            num_sample = 50
            pred_length = 100
            pred = obs[:, -2:-1, :, :, :].repeat(num_sample, pred_length, 1, 1, 1)
            return pred
    return DummyModel(), 'cpu'

model, device = create_model()
            
# prepare data loader
data_loader_name = "data_loader_test"
data_config_path = "compute_mmcm/default_parms/h36m/h36_config.json"  # or "compute_mmcm/default_parms/amass/amass_config.json"
data_config = read_json(data_config_path)
data_config = ConfigParser(data_config, save=False)
data_loader = prepare_data(data_config, data_loader_name, shuffle=False, drop_last=False, num_workers=0, batch_size=1, )
batch_size = data_loader.batch_size

store_folder = "baseline_output/dummy_model/h36m/"  # specify your output folder

# for each batch
batches_toenumerate = enumerate(data_loader)
for nbatch, batch in tqdm(batches_toenumerate, total=len(data_loader)):
    obs, target, extra = batch

    idces = extra["sample_idx"]
    obs, target = obs.to(device), target.to(device)

    pred = model(obs)
    # predictions -> [BS, nSamples, Seq_length, Partic, Joints, Feat]
    pred = data_loader.dataset.recover_landmarks(pred, rrr=False) # do not recover root, only denormalize if needed


    output_folder = os.path.join(store_folder, "npy")
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, f"{int(extra['sample_idx'])}_{extra['metadata'][0]}_{extra['metadata'][1]}_{int(extra['init'][0])}to{int(extra['end'][0])}_pred.npy"), pred[:,:,0,...].cpu())
    np.save(os.path.join(output_folder, f"{int(extra['sample_idx'])}_{extra['metadata'][0]}_{extra['metadata'][1]}_{int(extra['init'][0])}to{int(extra['end'][0])}_obs.npy"), obs.cpu())

        