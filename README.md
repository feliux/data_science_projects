# My Notebooks

This reposotory contains some of my big data & data science projects.

## Quick Start

### Environment

I usually work with docker containers, so...

- Install Docker

- Clone this repository

~~~
$ git clone https://github.com/feliux/data_science_projects.git
$ cd notebook
~~~

#### Spark

This notebooks runs over [jupyter/all-spark-notebook](https://hub.docker.com/r/jupyter/all-spark-notebook) image.

- Pull the image from Docker Hub

~~~
$ docker pull jupyter/all-spark-notebook
~~~ 

- Run your container

~~~
$ docker run -d --name <container_name> -v $PWD:/home/jovyan/work -p 8888:8888 <your_name>/all-spark-notebook:latest
~~~

Or execute

~~~
$ docker-compose -f docker-compose-spark.yml up -d
~~~

#### Pytorch

You can use the previous container for this notebooks working with CPU. If you want to use your **GPU** just pull or build a new image with the Dockerfile I provide (based on the [nvidia/cuda:10.2-base-ubuntu18.04](https://hub.docker.com/r/nvidia/cuda) image). Feel free to add/remove whatever you want ;P

- Pull from Docker Hub or build the image

~~~
$ docker pull feliux/gpu-scipy-pytorch-notebook
OR
$ docker build --no-cache=true --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') -t feliux/gpu-scipy-pytorch-notebook:1.0.0 Pytorch_Dockerfile/.
~~~

- Run your container. For example

~~~
$ docker run -d --gpus all --name gpu-pytorch -v $PWD:/home/jovyan/work -p 8888:8888 feliux/gpu-scipy-pytorch-notebook:1.0.0
~~~

Or execute

~~~
$ docker-compose -f docker-compose-pytorch.yml up -d
~~~

- Test installed packages (some examples)

~~~
$ docker exec -it gpu-pytorch python
>>> import torch; torch.cuda.is_available()
True
~~~

---

- Connect to your notebook

~~~
$ docker exec -t <container_name> jupyter-notebook list
~~~

- Paste the url/token in your favourite browser

## Folder structure

~~~
├── data -> datasets used on notebooks. on_time_performance_2016_12.csv dataset (spark) is not available cause it exceed 100MB permited on Github.
│   └── pytorch_data
│       └── MNIST
│           ├── processed
│           └── raw
├── images -> images used on notebooks.
├── pytorch -> deep learning notebooks. This notebooks imports pytorch with CUDA 10.2 compatibility (it was installed at Dockerfile).
└── spark -> machine learning with pyspark, sparkSQL, sparkml, sparkmllib.
~~~
