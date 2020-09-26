FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3

RUN echo "Build our Container based on L4T Pytorch"
RUN nvcc --version

# https://github.com/dusty-nv/jetson-containers/issues/5#issuecomment-632829664
COPY ./resources/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
COPY ./resources/jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

# install cmake (https://stackoverflow.com/a/55508937)
# packages for camera connection
# packages for display output
RUN apt-get update && apt-get install -y libopencv-python && apt-get install -y --no-install-recommends \
          python3-pip \
          python3-dev \
          build-essential \
          zlib1g-dev \
          zip \
          libxinerama-dev \
          libxcursor-dev \
          libjpeg8-dev \
          p7zip-full \ 
          x11-apps \ 
          xauth \ 
          vim \ 
          cmake \
          protobuf-compiler \ 
          xorg-dev \ 
          libglu1-mesa-dev \ 
          libusb-1.0-0-dev

RUN rm -rf /var/lib/apt/lists/*

# install face_recognition (https://github.com/ageitgey/face_recognition)
RUN pip3 install face_recognition setuptools Cython wheel numpy

WORKDIR /lib

# install torch2trt (https://github.com/NVIDIA-AI-IOT/torch2trt#setup)
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
WORKDIR /lib/torch2trt
RUN python3 setup.py install

WORKDIR /lib 

# https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md
RUN git clone https://github.com/IntelRealSense/librealsense
RUN mkdir librealsense/build
WORKDIR /lib/librealsense/build
RUN cmake ../ -DBUILD_PYTHON_BINDINGS=bool:true -DCMAKE_BUILD_TYPE=release -DBUILD_WITH_CUDA:bool=true
RUN make -j4 
RUN make install

# installation copies files to wrong location
# correct the location of some files 
RUN mv /usr/local/lib/python2.7/pyrealsense2 /usr/local/lib/python3.6/dist-packages
RUN mv /lib/librealsense/wrappers/python/pyrealsense2/__init__.py /usr/local/lib/python3.6/dist-packages/pyrealsense2

# copy resources from the folder
RUN mkdir app/ && cd app && mkdir IW276WS20-P6/
WORKDIR /app/IW276WS20-P6
COPY ./src/ ./src/
COPY ./pretrained-models ./pretrained-models 
RUN mkdir logs && mkdir models

# unzip model
WORKDIR /app/IW276WS20-P6/pretrained-models
RUN 7z x resnet50.224.pth.7z

WORKDIR /app/IW276WS20-P6/src

# convert2trt script always quits with an segmentation fault
# be carefull with the log output to spot errors beside the segmentation fault
RUN python3 convert2trt.py resnet50 ../pretrained-models/resnet50.224.pth ../models/resnet50.224.trt.pth; exit 0

# run the pipeline
CMD python3 pipeline.py ../models/resnet50.224.trt.pth
