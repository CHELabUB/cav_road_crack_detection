#!/bin/bash
# This bash script is used to launch a Docker container for the CV2X application on the Lincoln MKZ platform.
# For explanation of the tag, check official docker documentation:
# chelab/crack-detection:20250806 is the name of the docker image, which corresponds to the latest docker on the host machine. 
# The docker container will have a fixed name cv2x.
# The --rm flag will remove the container after it is stopped.
# The --privileged flag is used to give the container access to the host's devices.
#   -e DISPLAY=$DISPLAY \
#   -v ${XAUTHORITY:-$HOME/.Xauthority}:/root/.Xauthority  allows proper camera access with rviz2 inside docker

docker run -it --network host --runtime=nvidia --gpus all --privileged \
  -v /home/user/docker/seaslab/ws/:/ws \
  -v ${XAUTHORITY:-$HOME/.Xauthority}:/root/.Xauthority \
  --name=cv2x --rm \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  chelab/crack-detection:20250806
