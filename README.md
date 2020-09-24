# Project-Template for IW276 Autonome Systeme Labor

Short introduction to project assigment.

<p align="center">
  Screenshot / GIF <br />
  Link to Demo Video
</p>

> This work was done by Christian Braun, Fabian Wenzel and Bernardo Abreu Figueiredo during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

## Table of Contents

- [Project-Template for IW276 Autonome Systeme Labor](#project-template-for-iw276-autonome-systeme-labor)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Prerequisites](#prerequisites)
  - [Pre-trained models <a name="pre-trained-models"/>](#pre-trained-models-a-namepre-trained-models)
  - [Running](#running)
  - [Docker](#docker)
  - [Acknowledgments](#acknowledgments)
  - [Contact](#contact)

## Requirements
* Python 3.6 (or above)
* OpenCV 4.1 (or above)
* Jetson Nano
* Jetpack 4.4
* Docker
* librealsense
* 
> [Optional] ...
> 
- **requirements.txt schreiben!!!**

## Prerequisites
1. Install requirements:
```
pip install -r requirements.txt
```
- Kamera an Jetson Nano anschließen
  - Falls auf der Nano die Librealsense installiert ist demo laufen lassen (rs-depth, rs-enumerate)
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

## Pre-trained models <a name="pre-trained-models"/>

Pre-trained model is available at pretrained-models/
- ResNet 50 von P2

## Running

To run the demo, pass path to the pre-trained checkpoint and camera id (or path to video file):
```
python src/demo.py --model model/student-jetson-model.pth --video 0
```
> Additional comment about the demo.

- Laufen lassen der Applikation direkt über den Container bei CMD/Entrypoint
- oder über die bash shell im Container über 
- python3 pipeline.py

## Docker
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
