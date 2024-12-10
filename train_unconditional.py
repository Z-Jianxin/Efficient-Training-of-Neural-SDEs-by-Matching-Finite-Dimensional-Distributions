import argparse
import importlib.util
import sys
import torch
import torch.optim.swa_utils as swa_utils
from src.exp_helpers import get_data, train_model, adjust_generator_parameters, create_discriminator, setup_optimizers_and_schedulers
from src.gan.generators import Generator
from src.utils.helper_functions.global_helper_functions import mkdir
import os
from time import time
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    return args

def load_configurations(file_path):
    spec = importlib.util.spec_from_file_location("configurations", file_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["configurations"] = config_module
    spec.loader.exec_module(config_module)
    return config_module.config

def run_exp(config_file,):
    config = load_configurations(config_file)
    data_type, discriminator_type, device, seed = config["data_type"], config["discriminator_type"], config["device"], config["seed"]
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    ts, data_size, train_dataloader, test_dataloader, infinite_train_dataloader, transformer = get_data(data_type, config["data_config"], device)
    generator = Generator(data_size=data_size, **config["generator_config"]).to(device)
    generator = adjust_generator_parameters(
        generator,
        config["generator_config"],
        learning_type=config["data_config"]["learning_type"],
        **config["optimizer_config"]
    )
    discriminator = create_discriminator(discriminator_type, data_size, config[discriminator_type + "_config"]).to(device)
    averaged_generator     = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)

    generator_optimiser, discriminator_optimiser, g_scheduler, d_scheduler = setup_optimizers_and_schedulers(generator, discriminator, config["optimizer_config"])
    gen_fp = os.path.join(config["model_save_root"], f"generators/{data_type}_{discriminator_type}")
    disc_fp = os.path.join(config["model_save_root"], f"discriminators/{data_type}_{discriminator_type}")
    mkdir(os.path.join(config["model_save_root"], f"generators"))
    mkdir(os.path.join(config["model_save_root"], f"discriminators"))
    print(f"generator save path: {gen_fp}")
    print(f"discriminators save path: {disc_fp}")
    print(config["data_config"])
    print(config["optimizer_config"])
    print(config[discriminator_type + "_config"])
    tr_loss = train_model(
        config,
        device,
        generator,
        discriminator,
        generator_optimiser,
        discriminator_optimiser, 
        g_scheduler,
        d_scheduler,
        infinite_train_dataloader,
        transformer,
        ts,
        averaged_generator,
        averaged_discriminator, 
        gen_fp, 
        disc_fp
    )

if __name__ == "__main__":
    args = parse_arguments()
    start = time()
    run_exp(args.config_file)
    print(f"Training completes in {time()-start}s.")
