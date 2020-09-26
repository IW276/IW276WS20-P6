cp /etc/apt/sources.list.d/nvidia-l4t-apt-source.list .
cp /etc/apt/trusted.gpg.d/jetson-ota-public.asc .

sudo docker build . -t wenzeldock/asl-p6-pyrealsense2:3.2.0

rm nvidia-l4t-apt-source.list jetson-ota-public.asc