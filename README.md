# MMCM: Multimodality-aware Metric using Clustering-based Modes for Probabilistic Human Motion Prediction
This is the official repository for the following paper:

Anonymous

MMCM: Multimodality-aware Metric using Clustering-based Modes for Probabilistic Human Motion Prediction, 2025

![Top page](image/Overview.png)

## Environment

- Please install the appropriate version of PyTorch for your environment.
Then, install the remaining dependencies by running:
```
pip install -r requirements.txt
```

## Prepare datasets

- Prepare Human3.6M and AMASS following [BeLFusion](https://github.com/BarqueroGerman/BeLFusion) in "./auxiliar".

## Prepare weights and so on

- Download [weights set](), unzip the file, and place it in './compute_mmcm/default_parms'. (coming soon)

## Compute MMCM
### From numpy output
- You can evaluate predictions saved in NumPy format (.npy).
- The output results for several baseline methods can be downloaded from [coming soon](), and you unzip the file and place it in './baseline_output'.

```
# Baseline --> {comusion, belfusion, dlow}
# Dataset --> {h36m, amass}
python evaluate_baseline.py --pred_path "baseline_output/<Baseline>/h36m/npy/" --data_config_path "compute_mmcm/default_parms/<Dataset>/<Dataset>_config.json" --dataset_name <Dataset> --stride 25
```

### Base form
- coming soon


## Hyperparameter search
For example,
```
python compute_mmcm/parameter_search.py --data_config_path compute_mmcm/default_parms/h36m/h36_config.json --stride 25 --frames 103

python compute_mmcm/parameter_search.py --data_config_path compute_mmcm/default_parms/amass/amass_config.json --stride 60 --frames 123
```

