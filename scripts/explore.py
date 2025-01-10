## Simple Gym-like environment for KTH dataset
import numpy as np
import os 
import cv2
from tqdm.contrib.concurrent import process_map  # or thread_map
import time
from omegaconf import OmegaConf
import hydra 
import torch 
from torchvision.transforms import ToTensor, Resize, Compose
import pyastar2d    
import json
import multiprocessing
from tqdm import tqdm
import yaml
from torch.utils.data._utils.collate import default_collate
from skimage.measure import block_reduce
import albumentations as A
import traceback
import argparse
from pdb import set_trace as bp
from collections import deque
from queue import PriorityQueue
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# custom functions 
from lama_pred_utils import load_lama_model, visualize_prediction, get_lama_transform, convert_obsimg_to_model_input
# custom imports
import sys
sys.path.append('../')
from scripts.gen_building_utils import *
from scripts import simple_mask_utils as smu 
#from models.predictors.map_predictor_model import OccupancyPredictor
#from models.networks.unet_model import UNet
#import eval_deploy.viz_utils as vutils
#import eval_deploy.deploy_utils as dutils
#from eval_deploy import glocal_utils as glocal
import scripts.sim_utils as sim_utils
import upen_baseline
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_options_dict_from_yml(config_name):
    cwd = os.getcwd()
    hydra_config_dir_path = os.path.join(cwd, '../configs')
    print(hydra_config_dir_path)
    with hydra.initialize_config_dir(config_dir=hydra_config_dir_path):
        cfg = hydra.compose(config_name=config_name)
    options_dict = OmegaConf.to_container(cfg)
    options = OmegaConf.create(options_dict)
    return options

def run_exploration_comparison_for_map(args):
    map_folder_path = args['map_folder_path']
    models_list = args['models_list']
    lama_model = args['lama_model']
    lama_map_transform = args['lama_map_transform']
    pred_vis_configs = args['pred_vis_configs']
    lidar_sim_configs = args['lidar_sim_configs']
    start_pose = args['start_pose']
    modes_to_test = args['modes_to_test']
    unknown_as_occ = args['unknown_as_occ']
    use_distance_transform_for_planning = args['use_distance_transform_for_planning']
    upen_config  = args['upen_config']
    print("Running exploration comparison for map: ", map_folder_path)
    
    # Load in occupancy map and valid space map
    map_occ_npy_path = os.path.join(map_folder_path, 'occ_map.npy')
    assert os.path.exists(map_occ_npy_path), "Occupancy map path does not exist: {}".format(map_occ_npy_path)
    map_valid_space_npy_path = os.path.join(map_folder_path, 'valid_space.npy')
    assert os.path.exists(map_valid_space_npy_path), "Valid space map path does not exist: {}".format(map_valid_space_npy_path)
    occ_map, validspace_map = sim_utils.get_kth_occ_validspace_map(map_occ_npy_path, map_valid_space_npy_path)
    map_name = os.path.dirname(map_occ_npy_path)

    # Sample random start pose, if start_pose is None
    if start_pose is None:
        buffer_start_pose = 2
        start_pose = smu.sample_free_position_given_buffer(occ_map, validspace_map, buffer_start_pose)
        assert start_pose is not None, "Could not sample start pose"

    plt.imshow(occ_map)
    plt.scatter(start_pose[1], start_pose[0], c='r', s=10, marker='*')
    plt.title('GT Map (Red: Start Pose)')
    plt.savefig('gt_map.png')
    plt.close()
        
    # get overall experiment name
    # exp_title based on start time YYYYMMDD_HHMMSS, and name of map
    folder_name = os.path.basename(map_name)
    comp_exp_title = time.strftime("%Y%m%d_%H%M%S") + '_' + folder_name + '_' + str(start_pose[0]) + '_' + str(start_pose[1])
        
    for mode in modes_to_test:
        run_exploration_for_map(occ_map, comp_exp_title, models_list, lama_model, lama_map_transform, pred_vis_configs, lidar_sim_configs, mode, start_pose, unknown_as_occ, \
            use_distance_transform_for_planning=use_distance_transform_for_planning, upen_config=upen_config)

def run_exploration_for_map(occ_map, exp_title, models_list,lama_alltrain_model, lama_map_transform,  pred_vis_configs, lidar_sim_configs, mode, \
    start_pose, unknown_as_occ, use_distance_transform_for_planning, upen_config=None):
    pass

if __name__ == '__main__':
    data_collect_config_name = 'local.yaml' #customize yaml file, as needed
    output_subdirectory_name = '20250110_test' #customize subdirectory to save experiment results, as needed
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_world_list', nargs='+', help='List of worlds to collect data from')
    parser.add_argument('--start_pose', nargs='+', help='List of start pose')
    args = parser.parse_args()
    
    collect_opts = get_options_dict_from_yml(data_collect_config_name)
    if args.collect_world_list is not None:
        collect_opts.collect_world_list = args.collect_world_list
    if args.start_pose is not None: 
        start_pose = []
        for pose_elem in args.start_pose:
            start_pose.append(int(pose_elem))
        collect_opts.start_pose = start_pose

    kth_map_folder_path = os.path.join(collect_opts.root_path, '/kth_test_maps/')
    kth_map_paths = os.listdir(kth_map_folder_path)

    assert not (collect_opts.test_world_only and collect_opts.collect_world_list is not None), "Only one of test_world_only and collect_world_list can be true"
    kth_map_folder_paths = [os.path.join(kth_map_folder_path, p) for p in kth_map_paths] * collect_opts.num_data_per_world

    # Make output_subdirectory_name if it doesn't exist 
    output_root_dir = os.path.join(collect_opts.output_root_path, output_subdirectory_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    
    model_list = []
    device = collect_opts.lama_device

    #ensemble of lama models (G_i; G_1, G_2, G_3 in the paper) - fine-tuned with split, smaller training sets
    if collect_opts.ensemble_folder_name is not None:
        ensemble_folder_name = collect_opts.ensemble_folder_name
        ensemble_model_dirs = sorted(os.listdir(os.path.join(collect_opts.root_path, 'pretrained_models', ensemble_folder_name)))
        for ensemble_model_dir in ensemble_model_dirs:
            ensemble_model_path = os.path.join(collect_opts.root_path, 'pretrained_models', ensemble_folder_name, ensemble_model_dir)
            model = load_lama_model(ensemble_model_path, device=collect_opts.lama_device)
            print("Loaded model: ", ensemble_model_dir)
            model_list.append(model)
    
    #setup a big lama model (G in the paper) - fine-tuned with the entire training set
    lama_model = load_lama_model(collect_opts.big_lama_model_path, device=collect_opts.lama_device)
    lama_map_transform = get_lama_transform(collect_opts.lama_transform_variant, collect_opts.lama_out_size)

    run_exploration_args = []
    for kth_map_folder_path in kth_map_folder_paths:
        args_dict = {
            'map_folder_path': kth_map_folder_path,
            'models_list': model_list,
            'lama_model': lama_model,
            'lama_map_transform': lama_map_transform,
            'pred_vis_configs': collect_opts.pred_vis_configs,
            'lidar_sim_configs': collect_opts.lidar_sim_configs,
            'start_pose': collect_opts.start_pose,
            'modes_to_test': collect_opts.modes_to_test,
            'unknown_as_occ': collect_opts.unknown_as_occ,
            'use_distance_transform_for_planning': collect_opts.use_distance_transform_for_planning,
            'upen_config': collect_opts.upen_config
        }
        run_exploration_args.append(args_dict)
    
    for run_exploration_arg in run_exploration_args:
        run_exploration_comparison_for_map(run_exploration_arg)

