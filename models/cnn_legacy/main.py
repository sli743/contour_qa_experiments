import os
import sys
import yaml
import argparse
import shutil
import wandb
import random
from datetime import datetime
import torch
from utils.logger import Logger
from utils.helper import set_seed
from network.interface import model_train, model_test
import copy

def run_experiment(cf, alpha=None, kld_weight=None, weight_mode=None):
    cf_copy = copy.deepcopy(cf)
    if alpha is not None:
        cf_copy['alpha'] = alpha
    if kld_weight is not None:
        cf_copy['kld_weight'] = kld_weight
    if weight_mode is not None:
        cf_copy['weight_mode'] = weight_mode
    cf_copy['exp_name'] = f"{cf['exp_name']}_alpha{alpha}_kld{kld_weight}_wm{weight_mode}"
    cf_copy['output_folder'] = os.path.join(cf_copy['save_dir'], cf_copy['exp_name'])
    os.makedirs(cf_copy['output_folder'], exist_ok=True)
    
    # Setup logging for this experiment
    log_file = os.path.join(cf_copy['output_folder'], 'logfile_' + cf_copy['timestamp'] + '.log')
    sys.stdout = Logger(log_file)
    shutil.copyfile(args.config, os.path.join(cf_copy['output_folder'], 'config_' + cf_copy['timestamp'] + '.yaml'))
    
    print(f"\n > Running experiment: {cf_copy['exp_name']}")
    wandb.init(project="VAE-Variant-Radformation", entity="coral", name=cf_copy['exp_name'])
    wandb.config.update({
        'alpha': alpha,
        'kld_weight': kld_weight,
        'weight_mode': weight_mode,
        **{k: v for k, v in cf_copy.items() if k not in ['alpha', 'kld_weight', 'weight_mode']}
    })

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n > Utilized device: ' + str(device))
    if device.type == 'cuda':
        print('GPU: ' + torch.cuda.get_device_name(device))
    cf_copy['device'] = device

    # Training
    if cf_copy['train']:
        print('\n > Start the training process...')
        model_train(cf_copy)

    # Testing
    if cf_copy['test']:
        print('\n > Start the testing process...')
        model_test(cf_copy)

    wandb.finish()
    # Reset stdout for next experiment
    sys.stdout = sys.__stdout__

def main():
    global args
    parser = argparse.ArgumentParser(description='VAE Variants for Radformation Segmentation')
    parser.add_argument('-c', '--config', type=str, help='path to the config file.', required=True)
    args = parser.parse_args()

    # Load the config file
    with open(args.config) as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)

    # Set random seed
    seed = cf['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    set_seed(seed)
    print('\n > Utilized random seed: ' + str(seed))
    torch.backends.cudnn.allow_tf32 = cf['use_tf32']

    # Set base output folder and timestamp
    cf['timestamp'] = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Hyperparameter tuning
    alphas = cf.get('alpha', [0.9]) if isinstance(cf.get('alpha'), list) else [cf.get('alpha', 0.9)]
    kld_weights = cf.get('kld_weight', [1.0e-2]) if isinstance(cf.get('kld_weight'), list) else [cf.get('kld_weight', 1.0e-2)]
    weight_modes = cf.get('weight_mode', ['uniform']) if isinstance(cf.get('weight_mode'), list) else [cf.get('weight_mode', 'uniform')]
    # weight_modes = list(set(weight_modes) | {'uniform','sqrt_inverse_frequency'})  # Ensure all modes are included
    
    for alpha in alphas:
        for kld_weight in kld_weights:
            for weight_mode in weight_modes:
                run_experiment(cf, alpha=alpha, kld_weight=kld_weight, weight_mode=weight_mode)

if __name__ == '__main__':
    main()