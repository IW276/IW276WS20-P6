FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3

RUN echo "Build our Container based on L4T Pytorch"
RUN nvcc --version

# https://github.com/dusty-nv/jetson-containers/issues/5#issuecomment-632829664
COPY nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

RUN apt-get update

RUN apt-get install -y libopencv-python && apt-get install -y --no-install-recommends \
          python3-pip \
          python3-dev \
          build-essential \
          zlib1g-dev \
          zip \
          libxinerama-dev \
          libxcursor-dev \
          libjpeg8-dev

RUN apt-get update && apt-get -y install p7zip-full
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install setuptools Cython wheel
RUN pip3 install numpy --verbose

# install cmake and face_recognition (https://stackoverflow.com/a/55508937)
RUN apt-get update && apt-get -y install cmake protobuf-compiler

# install face_recognition (https://github.com/ageitgey/face_recognition)
RUN pip3 install face_recognition

# install yaml
RUN pip3 install pyyaml

# install torch2trt (https://github.com/NVIDIA-AI-IOT/torch2trt#setup)
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
WORKDIR /torch2trt
RUN python3 setup.py install

# clone old project and run pipeline
WORKDIR /

RUN apt-get update && apt-get -y install xorg-dev libglu1-mesa-dev libusb-1.0-0-dev

RUN git clone https://github.com/IntelRealSense/librealsense
RUN mkdir librealsense/build
WORKDIR /librealsense/build
RUN cmake ../ -DBUILD_PYTHON_BINDINGS=bool:true
RUN make -j4 VERBOSE=1 
RUN make install

RUN mv /usr/local/lib/python2.7/pyrealsense2 /usr/local/lib/python3.6/dist-packages
RUN mv /librealsense/wrappers/python/pyrealsense2/__init__.py /usr/local/lib/python3.6/dist-packages/pyrealsense2

RUN apt-get update && apt-get install -qqy x11-apps xauth vim

RUN mkdir IW276WS20-P6
WORKDIR /IW276WS20-P6
COPY ./src/ .
WORKDIR /IW276WS20-P6

ENTRYPOINT /bin/bash

# RUN python3 -c "import pyrealsense2"
# RUN git clone https://github.com/IW276/IW276WS20-P6.git

# WORKDIR /IW276WS20-P6
# RUN mv resources/pyrealsense2/ src/

# WORKDIR /IW276WS20-P6/src
# RUN python3 -c "import pyrealsense2"
# RUN python3 realsenseVideo.py

# RUN mkdir /IW276WS20-P6/models

# WORKDIR /IW276WS20-P6/pretrained-models
# RUN 7z x resnet50.224.pth.7z -o../models

# WORKDIR /IW276WS20-P6/models
# RUN ls

# WORKDIR /IW276WS20-P6/src
# RUN python3 convert2trt.py resnet50 ../models/resnet50.224.pth ../models/resnet50.224.trt.pth
# RUN python3 pipeline_main.py video.mp4

