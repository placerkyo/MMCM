import sys
import os
# 2つ上の階層を参照パスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import torch
import data_loader as module_data
import metrics.loss as module_loss
import metrics.metric as module_metric
import autoencoder.models as module_arch
from autoencoder.trainer import AutoencoderTrainer
from utils import prepare_device, read_json, add_dict_to_argparser, set_global_seed, update_config_with_arguments, compute_statistics
from parse_config import ConfigParser
from datetime import datetime
import random
import wandb

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"



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

def main(config_dict, resume):
    unique_id = datetime.now().strftime(r'%y%m%d_%H%M%S')  + f"_{random.randint(0,1000):03d}"
    config = ConfigParser(config_dict, resume=args.resume, run_id=unique_id)

    ## wandb
    if args.debug:
        mode = "disabled"
    else:
        mode = "online"
    wandb.init(
        project="mm_autoencoder",
        name=f"{config['name']}_{unique_id}",
        config=config.config,
        mode=mode,
    )

    seed = config["seed"]
    set_global_seed(seed)

    logger = config.get_logger('train')
    if resume:
        logger.info("---------------------- RESUMED ----------------------")

    data_loader = config.init_obj('data_loader_training', module_data)
    logger.info(f"Number of training samples: {data_loader.n_samples}")
    valid_data_loader = None
    if 'data_loader_validation' in config.config:
        valid_data_loader = config.init_obj('data_loader_validation', module_data)
        logger.info(f"Number of validation samples: {valid_data_loader.n_samples}")
    elif 'validation_split' not in config['data_loader_training']['args']: # no validation set, no validation split % set => no validation at all!
        logger.warning(f"Validation set was not loaded!")# Training will run for {epochs} epochs.")
        pass

    model = config.init_obj('arch', module_arch)
    if not resume:
        logger.info('Trainable parameters: {}'.format(model.get_params()))

    # prepare for (multi-device) GPU training
    for i in range(torch.cuda.device_count()):
        logger.info(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    samples_epoch = config['trainer']['samples_epoch'] if 'samples_epoch' in config['trainer'] else None
    valid_frequency = config['trainer']['validation_frequency']
    if valid_frequency > 0:
        logger.info(f"Running validation {valid_frequency} times per epoch.")
    else:
        logger.info(f"Validation is not activated.")

    es = config['trainer']['early_stop']
    assert (es not in (0,-1) and valid_frequency not in (0,-1)) or es in (0,-1), logger.error(f"Combination not possible: early_stop={es} and valid_frequency={valid_frequency}")
    

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss']["type"])
    criterion_params = dict(config['loss']['args'])
    
    metrics = []
    if "metrics" in config.config:
        for met in config['metrics']:
            assert "type" in met
            fn = getattr(module_metric, met["type"])
            metrics.append({
                'fn': fn,
                'alias': met.get('alias', fn.__name__),
                'params': dict(met.get('args', {}))
                })


    # ----------------------------------------------- TRAINING -----------------------------------------------
    trainer = AutoencoderTrainer(model, criterion, criterion_params, metrics, optimizer,
                    config=config,
                    device=device,
                    data_loader=data_loader,
                    valid_data_loader=valid_data_loader,
                    lr_scheduler=lr_scheduler, 
                    validation_frequency=valid_frequency,
                    samples_epoch=samples_epoch,
                    seed=seed,
                    wandb=wandb,
                 )
    last_epoch_stored = -1
    logger.info(f"Starting training (epochs={config['trainer']['epochs']}, early_stop={es})...")
    last_epoch_stored = trainer.train()
    logger.info(f"Training finished at epoch={last_epoch_stored}!")
    logger.info('=' * 80)

    # ----------------------------------------------- STATS COMPUTATION -----------------------------------------------
    logger.info(f"Starting stats computation...")
    config.resume = os.path.join(config._save_dir, f"checkpoint-epoch{last_epoch_stored}.pth")
    compute_statistics(config, "training")
    print("Stats computed!")
    logger.info('=' * 80)


def concat_jsons(json1, json2):
    # jsons are concatenated (merged) up to 2 levels in depth.
    for key in json1:
        if key in json2:
            for subkey in json1[key]:
                json2[key][subkey] = json1[key][subkey]
        else:
            json2[key] = json1[key]
    return json2


def create_argparser():
    """
    for key in defaults_to_config.keys():
        assert key in defaults, f"[code error] key '{key}' has no config path associated."
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--seed', default=DEFAULT_SEED, type=int,
                      help='random seed')
    parser.add_argument('-d', '--debug', action='store_true',
                      help='debug mode (default: False)')

    add_dict_to_argparser(parser, ARGS_TYPES)

    return parser


if __name__ == '__main__':
    args = create_argparser().parse_args()

    config_path = os.path.join(os.path.dirname(args.resume), "config.json") if args.resume else args.config
    config_dict = read_json(config_path)

    update_config_with_arguments(config_dict, args, ARGS_TYPES, ARGS_CONFIGPATH)
    config_dict["seed"] = args.seed
    config_dict["config_path"] = args.config

    main(config_dict, resume=args.resume is not None)

