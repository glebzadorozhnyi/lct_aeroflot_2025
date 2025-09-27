# FROM ubuntu:24.10
FROM ubuntu:22.04

RUN apt update -y && apt install python3 net-tools vim curl systemctl pip wget git -y

#INstall CUDA
#https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda-repo-ubuntu2204-12-5-local_12.5.0-555.42.02-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-12-5-local_12.5.0-555.42.02-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update --fix-missing
RUN apt-get -y install cuda-toolkit-12-5

#fixup missing ImportError: libGL.so.1 for cv2 (opencv-python)
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt update -y && apt-get install ffmpeg libsm6 libxext6  -y

# Needed by pycuda and tensorrt
ENV CUDA_ROOT=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# Install dependencies for python from requirements.txt file
WORKDIR /workdir
ADD requirements.txt /workdir/requirements.txt
RUN python3 -m pip install --upgrade pip wheel
RUN pip install -r requirements.txt

# https://github.com/THU-MIG/yolov10/blob/main/docker/Dockerfile
RUN pip install --no-cache nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0
RUN apt upgrade --no-install-recommends -y openssl tar
ADD https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt /usr/src/ultralytics/
RUN python3 -m pip install --upgrade pip wheel


COPY ./nix/install_nix.sh /tmp/install_nix.sh
RUN bash /tmp/install_nix.sh