# Project-Template for IW276 Autonome Systeme Labor

<a href="https://iwi-i-wiki.hs-karlsruhe.de/IWI_I/AutonomeSysteme/IW276WS20P6FaceExpressionRecognition"><img src="https://img.shields.io/badge/-Documentation-brightgreen"/></a>
<a href="https://hub.docker.com/repository/docker/wenzeldock/asl-p6-pyrealsense2"><img src="https://img.shields.io/badge/-Docker Hub-blue"/></a>

Short introduction to project assigment.

<p align="center">
  <img src="./demo.gif" />
</p>

> This work was done by Christian Braun, Fabian Wenzel and Bernardo Abreu Figueiredo during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

## Table of Contents

- [Project-Template for IW276 Autonome Systeme Labor](#project-template-for-iw276-autonome-systeme-labor)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Prerequisites](#prerequisites)
    - [Docker Execution Prerequisites](#docker-execution-prerequisites)
    - [Nano Execution Prerequisites](#nano-execution-prerequisites)
      - [**torch2trt**](#torch2trt)
      - [**pyrealsense2 (librealsense)**](#pyrealsense2-librealsense)
  - [Pre-trained models <a name="pre-trained-models"></a>](#pre-trained-models-)
  - [Running](#running)
  - [Docker](#docker)
  - [Acknowledgments](#acknowledgments)
  - [Contact](#contact)

## Requirements

* Jetson Nano
* Jetpack 4.4
* Docker 19.03 (or above)
* Python 3.6 (or above)
* OpenCV 4.1 (or above)
* numpy 1.19.1
* torch 1.6.0
* torchvision 0.7
* face_recognition 1.3.0
* torch2trt (see [Nano Prerequisites](#torch2trt))
* pyrealsense2 2.38.1.2225 (see [Nano Prerequisites](#pyrealsense2-librealsense))

## Prerequisites

Ensure that the camera is connected to the Jetson Nano. 
To check if the camera is connected correctly, you can run `rs-depth` or `rs-enumerate`. These commands are avaiable, **after** the installation of the pyrealsense2 library.

### Docker Execution Prerequisites

1. Clone the repository (https **or** ssh)
```
// https
git clone https://github.com/IW276/IW276WS20-P6.git

// ssh
git clone git@github.com:IW276/IW276WS20-P6.git
```
2. Move inside the directory
```
cd IW276WS20-P6
```

3. Copy Source List and GPG Files

```
cp /etc/apt/sources.list.d/nvidia-l4t-apt-source.list .
cp /etc/apt/trusted.gpg.d/jetson-ota-public.asc .
```
These files are needed to install OpenCV via apt-get inside the docker container.  
[Read the comment for more information](https://github.com/dusty-nv/jetson-containers/issues/5#issuecomment-632829664)

To build and run the docker container follow [here](#docker)

### Nano Execution Prerequisites

- Install all required dependencies from the [`requirements.txt`](./requirements.txt)

```
pip3 install -r requirements.txt
```

Some dependencies are not available via pip. The following sections guide the installation.

#### **torch2trt**

torch2trt is needed for the conversion of the pytorch model to the tensorRT model and the execution of the converted model.
torch2trt at this stage is not available via pip.
You need to clone the [repository](https://github.com/NVIDIA-AI-IOT/torch2trt/) and follow the [setup](https://github.com/NVIDIA-AI-IOT/torch2trt/#setup).

Setup:
```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

#### **pyrealsense2 (librealsense)**

The pip version of the library is not available for devices with arm architecutre. So we need to install the library from the repository.

https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md
```
git clone https://github.com/IntelRealSense/librealsense
mkdir librealsense/build
cd librealsense/build
cmake ../ -DBUILD_PYTHON_BINDINGS=bool:true -DCMAKE_BUILD_TYPE=release -DBUILD_WITH_CUDA:bool=true
make -j4 VERBOSE=1 
make install
```

You can check whether the installation was successfull with:
```
python3 -c "import pyrealsense2"
```

It can occur that the module cannot be found.
The installation command `make install` fails to copy needed files/scripts to the correct python 3.6 library folder. 
The location for the **pyrealsense2** folder can vary for each device / operating system.

In general two additional steps need to be done:
* Copy the `pyrealsense2` folder into the correct python 3.x library folder
* Copy the missing `__init__.py` file into the pyrealsense2 folder 

For example the steps on the Jetson Nano look like this:

```
mv /usr/local/lib/python3.6/pyrealsense2 /usr/local/lib/python3.6/dist-packages
mv /librealsense/wrappers/python/pyrealsense2/__init__.py /usr/local/lib/python3.6/dist-packages/pyrealsense2
```

## Pre-trained models <a name="pre-trained-models"></a>

Pre-trained model is available at pretrained-models/
- ResNet 50 von P2

## Running

Before running the scirpts directly on the nano (without docker) you need to successfully done the [nano prerequisites](#nano-execution-prerequisites).

To run the demo, pass path to the pre-trained checkpoint and camera id (or path to video file):
```
python3 pipeline.py ../models/resnet50.224.trt.pth
```
> Additional comment about the demo.

- Laufen lassen der Applikation direkt über den Container bei CMD/Entrypoint
- oder über die bash shell im Container über

## Docker

Before building and running the container you need to successfully done the [docker prerequisites](#docker-execution-prerequisites).


4. 

- Mehrere Möglichkeiten:
  - Docker Image selber bauen (ca. 40min bis 1h)
    - Wenn selber gebaut, Name im docker-compose.yml
  - Gebautes Docker Image aus der Registry herunterladen
    - docker-compose pull bzw. docker pull wenzeldock/asl-face-recoginiton
- Docker-compose ausführen  
  - mehrere möglichkeiten:
    - selber im container das skript ausführen
    - das skript direkt bei start des containers ausführen
- Konvertierung des Models
  - Falls nicht das Model im Folder models verwendet werden soll
  - oder eine neue Pytorch Version verwendet wird als 1.6 und diese inkompatibel mit der bisherigen Version ist
  - muss diese gebaut werden und im models ordner platziert werden
  - das docker-compose.yml erstellt eine volume, sodass man auf die models ordner im container zugreifen kann

HOW TO

- HOW TO was? 
  - Befehle zum Ausführen des Docker Containers?
  - Installieren der Dependencies für die Applikation?

## Acknowledgments

This repo is based on
  - [Source 1](https://github.com/)
  - [Source 2](https://github.com/)

- P2 erwähnen 
- librealsense beispiel für alignement und segmentation
- multithreading mit queue 
- erklärungen um den docker container auf der nano korrekt bauen zu können

Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
