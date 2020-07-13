# WELCOME TO DEEPNANO DEMO

### Index
[[_TOC_]]

## License

Usage of the provided software implies agreement with the terms of the [license](https://gitlab.kaust.edu.sa/makam0a/deepnano/-/blob/master/LICENSE) document available in this repository.
Briefly, the use of this software for commertial applications is not permitted and proper credit must be given whenever this 
software is used to produce results for publications.

## Overall description

This repository contains a demonstration of the flat optics design software ALFRED described in detail in the publication: 

*Broadband vectorial ultra-flat optics with experimental efficiency up to 99% in the visible via universal approximators*

currenlty under review but presently available as an [arxiv preprint](https://arxiv.org/abs/2005.01954). The code makes use of the theory described in the publication:

*Generalized Maxwell projections for multi-mode network Photonics* [Scientific Reports volume 10, Article number: 9038 (2020)](https://doi.org/10.1038/s41598-020-65293-6)

The user is ecouraged to read these publications to familiarize themselves with the underlying theory and logic behind the provided software.

#### Alfred
ALFRED stands for Autonomous Learning Framework for Rule-based Evolutionary Design, it is an inverse design software platform 
intended for the design of high efficiency flat optics. Given a desired optical response as the input ALFRED will find the
nanoscale geometry of the device that best approximates this response. 

The program is composed of two parts: A particle swarm optimizer and a neural network prediction unit

![Alfred_image](https://gitlab.kaust.edu.sa/makam0a/deepnano/-/raw/assets/alfred_overview.png)

Alfred works by launching the particles into a multidimensional search space containing a very large number of possible
nanostructure geometries. Each particle evaluates the peformance of a candidate geometry and explores the search space
according to the the values assigned to its inertia, social and memory paremeters. The behaviour of the particles is
intended to resemble the behaviour of social insects, such as ants or bees, in the sense that the exploration of the 
environment is carried out by indidivuals that share information with each other. For fast evaluation of the performance
of a possible candidate geometry each particle is equipped with a neural network prediction unit. The unit has been trained
on a set of FDTD simulations to be able to quickly and accurately predict the optical response of candidate geometries.  
Structurally, the predictor consists of the combination of a convolutional neural netork (CNN) based on the ResNet18 architecture 
and a series of fully connected networks (FCN) at the output. The CNN extracts features from an image representing the candidate
geometry and feeds this information to one of the FCNs, which returns the predicted optical response. The choice of FCN depends
on the thickness of the candidate structure, as each FCN has been trained using a specific thickness value.

In a typical search scenario ALFRED begins by launching a swarm of particles equipped with predictor units to quickly explore
the solution space. Once the particles converge to a candidate solution, ALFRED launches a second set of particles around it 
but with the predictor unit removed. These particles then execute full FDTD simulations to refine the candidate into the final
solution structure.

## Limitations of the provided software

The version of ALFRED as provided here is fully capable of returning nanostructure geometries for approximating a given 
input optical response. However, the code in this repository is only intended to provide a demonstration of the platform
operation to validate the results of its associated academic publication. As this software can be used to produce commertializable
devices, in order to protect the financial interests of the authors the final optimization routine has been removed. Any interested
parties who which to use this software with full optimizations for commertial applications can contact the authors to work out a licensing agreement.


# Getting started

## Requierements

### Hardware

The codes provided are optimized for running on an CUDA capable NVIDIA GPU.
While not strictly requiered, the user is advised that the neural network training
process can take several hours when running on the GPU and may become prohibitibly
long if running on a CPU. 

### Software

The use of a Linux based operating system is strongly recommended. 
All codes were tested on a Ubuntu 18.10 system and cross platform compatility 
is not guaranteed.

A working distribution of python 3.8 or higher is requiered to run the provided codes.
The use of the [anaconda python distribution](https://www.anaconda.com/) is recommended
to ease the installation of the requiered python packages.

The usage examples of this software are provided as Jupyter notebooks and as such 
requiere the [Jupyter notebook](https://jupyter.org/) package to run. Note this package
is included by default in the anaconda distribution.


## Initial set up

The usage of an Ubuntu 18.10 system with a CUDA capable GPU and the anaconda python
distribution is assumed for the rest of this document. 

### Obtaining the code

Begin by cloning this project along with the auxillary nanocpp-extras package to your local workstation.

```sh
$ git clone https://gitlab.kaust.edu.sa/primalight/deepnano
```

### Obtaining the dataset

A large (1.6 GB) dataset for training ALFRED is maintained as a compressed zip file [here](https://drive.google.com/uc?export=download&id=1i4V2YO8q1otAXowoV-r2rxgCK1rPe0Pf)

From the terminal, you can download this dataset using the python utility [gdown](https://github.com/wkentaro/gdown)

```bash
$ pip install gdown
$ gdown https://drive.google.com/uc?export=download&id=12chjLRBOMoQEf3voPFToDHEOOt1uLkmZ
```

To extract the zip file, the following command may be used

```bash
$ python -c "from zipfile import PyZipFile; PyZipFile( '''alfred_data.zip''' ).extractall()";
```


### System setup

The use of a separate python virtual environment is recommended for running the provided
programs. The file "deepnano.yml" is provided to quickly setup this environment. To create an environment
using the provided file do

```bash
$ cd deepnano
$ conda env create -f deepnano.yml
$ conda activate tensorflow
```

The code assumes that paths to the project and data folders exist in the system's
environmental variables. To set up these variables execute the following commands

```bash
$ export DATADIR="path-to/data/"
$ export PROJECT="path-to/project/"
```

The lines above can be added to your .bashrc file or equivalent so they are maintained
between system reboots.



## Usage

Usage instructions are provided in the jupyter notebook files of the repository. The user is adviced to first go through the 
file 'Demo.ipynb' as it explains how ALFRED handles data, the training process of the predictor and how to replicate the results
of the manuscript. The notebook can be viewed by executing the following commands:

```bash
$ jupyter notebook Demo.ipynb
```
## Citing

When making use of the provided codes in this repository for your own work please ensure you reference the [original publication](https://arxiv.org/abs/2005.01954). 
The following biblatex entry on the preprint is provided for your convenience while the final article undergoes review.

```
@article{Getman2020,
  title = {Broadband Vectorial Ultra-Flat Optics with Experimental Efficiency up to 99\% in the Visible via Universal Approximators},
  author = {Getman, Fedor and Makarenko, Maksim and Burguete-Lopez, Arturo and Fratalocchi, Andrea},
  date = {2020-05-05},
  url = {http://arxiv.org/abs/2005.01954},
  urldate = {2020-05-11},
  archivePrefix = {arXiv},
  eprint = {2005.01954},
  eprinttype = {arxiv},
  keywords = {Physics - Optics},
  primaryClass = {physics}
}
```
