# ---sources---
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/9.2/ubuntu18.04-x86_64/base/Dockerfile
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/9.2/ubuntu18.04-x86_64/runtime/Dockerfile
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/9.2/ubuntu18.04-x86_64/runtime/cudnn7/Dockerfile
# https://hub.docker.com/r/activeeon/cuda/dockerfile

FROM debian:buster

LABEL maintainer "jxtrbtk <yeah at free.fr>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64//7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1710/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 9.2.148
ENV CUDA_PKG_VERSION 9-2=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-$CUDA_PKG_VERSION \
    && ln -s cuda-9.2 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.2"

ENV NCCL_VERSION 2.4.8

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-$CUDA_PKG_VERSION \
    cuda-npp-$CUDA_PKG_VERSION \
    cuda-nvtx-$CUDA_PKG_VERSION \
    libnccl2=$NCCL_VERSION-1+cuda9.2 \
    && apt-mark hold libnccl2 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDNN_VERSION 7.6.5.32

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda9.2 \
    && apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN apt-get clean && apt-get update -y -qq

# Install python
RUN apt-get install -y python3 python3-pip python3-dev python3-setuptools

# Install generic packages
RUN apt-get install -y \
        build-essential cmake pkg-config wget \
        swig \
        git curl unzip nano \
        libaio1 \
        freetds-dev \ #MS SQL / Sybase over TCP/IP
        libjpeg-dev \
        libtiff5-dev \
        libpng-dev \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libatlas-base-dev \
        gfortran \
        libtbb2 \
        libtbb-dev \
        libgl1-mesa-glx

RUN apt-get install openmpi-bin -y

# Install python & pip
RUN rm /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN pip install setuptools --upgrade

# I love PyTorch
RUN pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# install dependancies 
RUN pip install deap update_checker tqdm stopit
RUN pip install scipy 
RUN pip install numpy 
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install opencv-python

# install most boost trees 
RUN pip install xgboost 
RUN pip install catboost
RUN pip install lightgbm

# TPot
RUN pip install tpot

# cleaning
RUN rm -rf ~/.cache/pip
RUN apt-get autoclean && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD bash


