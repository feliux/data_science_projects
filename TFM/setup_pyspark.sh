#!/bin/bash

ANACONDA_FILE=Anaconda3-2020.02-Linux-x86_64.sh
ANACONDA_LOCAL_FILE=/tmp/$ANACONDA_FILE
ANACONDA_REMOTE_FILE=https://repo.anaconda.com/archive/$ANACONDA_FILE
ANACONDA_MD5=17600d1f12b2b047b62763221f29f2bc

# Download and install Anaconda
CONDA_VERSION=$(conda --version 2> /dev/null)
if [ "$CONDA_VERSION" != "conda 4.2.9" ]
then
  curl -o $ANACONDA_LOCAL_FILE $ANACONDA_REMOTE_FILE
  FILE_MD5=$(md5sum $ANACONDA_LOCAL_FILE | awk '{print $1}')
  if [ "$ANACONDA_MD5" != "$FILE_MD5" ]
  then
      echo "Download of file: $ANACONDA_FILE failed, try again"
      rm $ANACONDA_FILE
      exit 1
  fi
  bash $ANACONDA_LOCAL_FILE
  rm -f $ANACONDA_LOCAL_FILE

  # Add environment variables to .bashrc
  if ! grep -Fq "PYSPARK_DRIVER_PYTHON=" ~/.bashrc
  then
    echo 'export PYSPARK_DRIVER_PYTHON=jupyter' >> ~/.bashrc
  fi
  if ! grep -Fq "PYSPARK_DRIVER_PYTHON_OPTS=" ~/.bashrc
  then
    echo "export PYSPARK_DRIVER_PYTHON_OPTS='notebook --allow-root --ip=0.0.0.0 --port=55489 --notebook-dir=/home/cloudera/galaxias'" >> ~/.bashrc
  fi
else
  echo "Anaconda already installed at the convenient version"
fi

# Source .bashrc to set environement variables and add Anaconda binaries to path
. ~/.bashrc