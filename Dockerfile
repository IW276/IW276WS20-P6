FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3

RUN echo "Build our Container based on L4T Pytorch"
RUN nvcc --version

RUN mkdir app

# COPY requirements.txt /app
COPY  nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
COPY  jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

RUN apt-get update

#Example:
RUN apt-get install -y libopencv-python && apt-get install -y --no-install-recommends \
          python3-pip \
          python3-dev \
          build-essential \
          zlib1g-dev \
          libssl-dev \
          zip \
          libjpeg8-dev && rm -rf /var/lib/apt/lists/*

RUN pip3 install setuptools Cython wheel
RUN pip3 install numpy --verbose

#install cmake and face_recognition
RUN apt-get update && apt-get -y install cmake protobuf-compiler
RUN pip3 install face_recognition

#clone torch2trt
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt

WORKDIR torch2trt
#install torch2trt
RUN python3 setup.py install