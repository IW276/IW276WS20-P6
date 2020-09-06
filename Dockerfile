FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3

RUN echo "Build our Container based on L4T Pytorch"
RUN nvcc --version

# https://github.com/dusty-nv/jetson-containers/issues/5#issuecomment-632829664
COPY nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

RUN apt-get update

# https://github.com/dusty-nv/jetson-containers/issues/5#issuecomment-673458718
RUN apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
RUN mv jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

RUN apt-get install -y libopencv-python && apt-get install -y --no-install-recommends \
          python3-pip \
          python3-dev \
          build-essential \
          zlib1g-dev \
          zip \
          libxinerama-dev \ 
          libxcursor-dev \
          libjpeg8-dev 
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install setuptools Cython wheel
RUN pip3 install numpy --verbose

# install cmake and face_recognition (https://stackoverflow.com/a/55508937)
RUN apt-get update && apt-get -y install cmake protobuf-compiler

# install face_recognition (https://github.com/ageitgey/face_recognition)
RUN pip3 install face_recognition

# install torch2trt (https://github.com/NVIDIA-AI-IOT/torch2trt#setup)
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
WORKDIR /torch2trt
RUN python3 setup.py install