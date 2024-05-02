# ML Lab

This reposotory contains some of my projects related to machine learning & data science.

## Quick Start

### Environment

I usually work with docker containers, so...

- Install Docker

- Clone this repository

```sh
$ git clone https://github.com/feliux/data_science_projects.git
$ cd notebook
```

#### Spark

Notebooks runs over [jupyter/all-spark-notebook](https://hub.docker.com/r/jupyter/all-spark-notebook) image.

- Pull the image from Docker Hub

```sh
$ docker pull jupyter/all-spark-notebook
``` 

- Run your container

```sh
$ docker run -d --name <container_name> -v $PWD:/home/jovyan/work -p 8888:8888 <your_name>/all-spark-notebook:latest
```

Or execute

```sh
$ docker-compose -f docker-compose-spark.yml up -d
```

#### Pytorch

You can use the previous container for this notebooks working with CPU. If you want to use your **GPU** just pull or build a new image with the Dockerfile I provide (based on the [nvidia/cuda:10.2-base-ubuntu18.04](https://hub.docker.com/r/nvidia/cuda) image). Feel free to add/remove whatever you want ;P

- Pull from Docker Hub or build the image

```sh
$ docker pull feliux/gpu-scipy-pytorch-notebook
OR
$ docker build --no-cache=true --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') -t feliux/gpu-scipy-pytorch-notebook:1.0.0 Pytorch_Dockerfile/.
```

- Run your container. For example

```sh
$ docker run -d --gpus all --name gpu-pytorch -v $PWD:/home/jovyan/work -p 8888:8888 feliux/gpu-scipy-pytorch-notebook:1.0.0
```

Or execute

```sh
$ docker-compose -f docker-compose-pytorch.yml up -d
```

- Test installed packages (some examples)

```sh
$ docker exec -it gpu-pytorch python
>>> import torch; torch.cuda.is_available()
True
```

- Connect to your notebook

```sh
$ docker exec -t <container_name> jupyter-notebook list
```

- Paste the url/token in your favourite browser

## References

[Introduction to Hidden Markov Models with Python Networkx and Sklearn](http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017)

[Hidden Markov Model](https://medium.com/@kangeugine/hidden-markov-model-7681c22f5b9)

[Training the Perceptron with Scikit-Learn and TensorFlow](https://www.quantstart.com/articles/training-the-perceptron-with-scikit-learn-and-tensorflow/)

[Graph Neural Networks](https://gnn.seas.upenn.edu/labs/lab1/)

[Python Speed](https://pythonspeed.com/)

[CNN](https://poloclub.github.io/cnn-explainer/)

[CNN](https://www.aprendemachinelearning.com/como-funcionan-las-convolutional-neural-networks-vision-por-ordenador/)
