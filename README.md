<p align="center">
<h1 align="center">MapEx: Indoor Structure Exploration with Probabilistic Information Gain from Global Map Predictions</h1>
<h3 class="is-size-5 has-text-weight-bold" style="color: orange;" align="center">
    Venue
</h3>
  <p align="center">
    <a href="https://cherieho.com/"><strong>Cherie Ho*</strong></a>
    ·
    <a href="https://cherieho.com/"><strong>Seungchan Kim*</strong></a>
    .
    <a href="https://cherieho.com/"><strong>Brady Moon</strong></a>
    .
    <a href="https://cherieho.com/"><strong>Aditya Parandekar</strong></a>
    .
    <a href="https://cherieho.com/"><strong>Narek Harutunyan</strong></a>
    <br>
    <a href="https://sairlab.org/team/chenw/"><strong>Chen Wang</strong></a>
    ·
    <a href="https://www.cs.cmu.edu/~./katia/"><strong>Katia Sycara</strong></a>
    ·
    <a href="https://theairlab.org/team/sebastian/"><strong>Graeme Best</strong></a>
    .
    <a href="https://theairlab.org/team/sebastian/"><strong>Sebastian Scherer</strong></a>
    <br>
  </p>

</p>



## Table of Contents


### Install using pip
You can install all requirements using pip by running:

    pip install -r mapper/requirements.txt

### Use Docker
To use Mapper using Docker, please follow the steps:
1. Build the docker image `mapper/Dockerfile` by running: 
        
        cd mapper/
        docker build -t mapper:release mapper

2. Launch the container while mounting this repository to the container file system.
    
        docker run -v <PATH_TO_THIS_REPO>:/home/mapper --network=host -it --gpus=all mapper:release


## Trained Weights

## License

## Acknowledgement


## Citation

If you find our paper, dataset or code useful, please cite us:

```bib
@article{ho2024mapex,
  title={MapEx: Indoor Structure Exploration with Probabilistic Information Gain from Global Map Predictions},
  author={Ho, Cherie and Kim, Seungchan and Moon, Brady and Parandekar, Aditya and Harutyunyan, Narek and Wang, Chen and Sycara, Katia and Best, Graeme and Scherer, Sebastian},
  journal={arXiv preprint arXiv:2409.15590},
  year={2024}
}
```