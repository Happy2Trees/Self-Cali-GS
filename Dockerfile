FROM nvcr.io/nvidia/pytorch:24.12-py3 AS base

ARG COLMAP_GIT_COMMIT=3.11.1
ARG CUDA_ARCHITECTURES=all-major
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT} && \
    git checkout FETCH_HEAD && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap-install && \
    ninja install



FROM nvcr.io/nvidia/pytorch:24.12-py3 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Limit CUDA archs for PyTorch extensions to common GPUs
# PyTorch expects dotted compute capabilities (e.g., 7.0, 8.6)
# ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

######### install colmap

RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev

######## PIP
# OS 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    screen \
    tzdata \
    ffmpeg \
    curl \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# 타임존을 기본값으로 설정 (UTC 예시)
RUN echo "Etc/UTC" > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata


WORKDIR /temp
COPY 3dgs-pose /temp/3dgs-pose
RUN pip install -e ./3dgs-pose

WORKDIR /temp
COPY simple-knn /temp/simple-knn
RUN pip install -e ./simple-knn

WORKDIR /temp
COPY requirements.txt /temp/requirements.txt
RUN pip install -r requirements.txt


COPY --from=base /colmap-install/ /usr/local/
CMD ["/bin/bash"]
