<p align="center">
<h1 align="center">MapEx: Indoor Structure Exploration with Probabilistic Information Gain from Global Map Predictions</h1>
<h3 class="is-size-5 has-text-weight-bold" style="color: orange;" align="center">
    Venue
</h3>
  <p align="center">
    <a href="https://cherieho.com/" target="_blank"><strong>Cherie Ho*</strong></a>
    ·
    <a href="https://seungchan-kim.github.io" target="_blank"><strong>Seungchan Kim*</strong></a>
    ·
    <a href="https://bradymoon.com/" target="_blank"><strong>Brady Moon</strong></a>
    ·
    <a href=""><strong>Aditya Parandekar</strong></a>
    ·
    <a href=""><strong>Narek Harutyunyan</strong></a>
    <br>
    <a href="https://sairlab.org/team/chenw/" target="_blank"><strong>Chen Wang</strong></a>
    ·
    <a href="https://www.cs.cmu.edu/~./katia/" target="_blank"><strong>Katia Sycara</strong></a>
    ·
    <a href="https://profiles.uts.edu.au/Graeme.Best" target="_blank"><strong>Graeme Best</strong></a>
    ·
    <a href="https://theairlab.org/team/sebastian/" target="_blank"><strong>Sebastian Scherer</strong></a>
    <br>
  </p>
</p>
  <h3 align="center"><a href="https://arxiv.org/abs/2409.15590">Paper</a> | <a href="">Project Page</a> | <a href="">Video</a></h3>
  <div align="center"></div>


## Table of Contents

### Set up Mamba environment
Mamba is a package manger used for managing python environments and dependencies, known for having better speed and efficiency than conda. For more information, please refer to this <a href="https://mamba.readthedocs.io/en/latest/user_guide/mamba.html">link</a>. 

    wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
    bash Mambaforge-Linux-x86_64.sh

Go to `lama` submodule folder, and create `lama` environment. 

    cd ~/MapEx/lama
    mamba env create -f conda_env.yml
    mamba activate lama

Install `torch` and relevant packages

    mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch --y
    mamba install wandb --yes
    pip install pytorch-lightning==1.2.9

### Download pretrained prediction models (KTH dataset)
You can download pretrained models from this <a href="https://drive.google.com/drive/u/0/folders/1u9WZ9ftwaMbP-RVySuNSVEdUDV_x4Dw6">link</a>. Place the zip file under `pretrained_models` directory and unzip the file. 

    mv ~/Downloads/weights.zip ~/MapEx/pretrained_models/
    cd ~/MapEx/pretrained_models/
    unzip weights.zip

The `pretrained_model` directory and its subdirectories should be organized as below: 

    MapEx
    ├── pretrained_models
        ├── weights
            ├── big_lama
                ├── models
                    ├── best.ckpt
            ├── lama_ensemble
                ├── train_1
                    ├── models
                        ├── best.ckpt
                ├── train_2
                    ├── models
                        ├── best.ckpt
                ├── train_3
                    ├── models
                        ├── best.ckpt    

### Install lib_rangec
Go to `range_libc` directory and install by following:

    cd ~/MapEx/lib_rangec/pywrapper
    python3 setup.py install 
    pip3 install Cython

### Install KTH toolbox dependencies for raycasting and observation model

    python3 -m pip install pyastar2d
    mamba install numba --yes

### Experiments (Running MapEx, baselines, and ablations)
In order to test MapEx, run the `explore.py` script. 

    cd scripts/
    python3 explore.py

This will automatically call `base.yaml`, which contains default parameters and specifies filepaths, environment, and starting conditions. If you want to customize parameters, generate your own yaml file and save it in the `configs` directory. 

Moreover, if you want to specifiy environment and starting positions as arguments to the script. For example, 

    python3 explore.py --collect_world_list 50010535_PLAN1 --start_pose 768 551

The list of environments is in the `kth_test_maps` directory. The list of start_pose for each environment can be seen in the `clusters/run_explore.job`. 

#### Details on methods 
`modes_to_test` in the yaml file specifies the methods that you test. Specifically,  

    modes_to_test: ['visvarprob'] #if you just want to test MapEx
    modes_to_test: ['nearest', 'upen', 'hectoraug', 'visvarprob'] #if you want to compare MapEx and baselines
    modes_to_test: ['obsunk', 'onlyvar', 'visunk', 'visvar', 'visvarprob'] #if you want ablation experiments
    modes_to_test: ['nearest', 'obsunk', 'onlyvar', 'visunk', 'visvar', 'visvarprob', 'upen', 'hector', 'hectoraug'] #if you want to test all methods 

`visvarprob` corresponds to the `MapEx` method, meaning the combination of visibility mask, variance, and probabilistic raycast. `nearest` is nearest frontier-based exploration, `upen` is our implementation of uncertainty-driven planner, proposed by <a href="https://arxiv.org/abs/2202.11907">Georgakis et al. ICRA 2022</a>, and `hectoraug` is our implementation of IG-hector method proposed by <a href="https://ieeexplore.ieee.org/document/8793769">Shrestha et al. ICRA 2019</a>.

<strong>Ablated methods</strong>: `visvar` (visibility mask + variance + deterministic raycast), `visunk` (visibility mask + counting number of pixels in the area), `obsunk` (visibility mask on observed occupancy grid + counting number of pixels in the area), `onlyvar` (using no visibility mask, but only summing variances) correspond with Deterministic, No Variance, Observed Map, and No Visibility methods in the ablation studies section of our original paper. 

## Citation

If you find our paper or code useful, please cite us:

```bib
@article{ho_kim2024mapex,
  title={MapEx: Indoor Structure Exploration with Probabilistic Information Gain from Global Map Predictions},
  author={Ho, Cherie and Kim, Seungchan and Moon, Brady and Parandekar, Aditya and Harutyunyan, Narek and Wang, Chen and Sycara, Katia and Best, Graeme and Scherer, Sebastian},
  journal={arXiv preprint arXiv:2409.15590},
  year={2024}
}
```