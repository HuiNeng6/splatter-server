FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
ARG TARGETPLATFORM TARGETARCH TARGETOS
# Install Python 3.12 and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        python3.12 \
        python3.12-venv \
        python3-pip \
        git \
        curl \
        zip \
        unzip \
        gpg \
        wget \
        tar \
        nano \
        gcc-14 \
        g++-14 \
        gfortran-14 \
        ninja-build \
        pkg-config \
        libxinerama-dev \
        libxcursor-dev \
        xorg-dev \
        libglu1-mesa-dev \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# LichtFeld needs CMake 3.30+
# Install it from Kitware repository (Ubuntu's apt has 3.28 which is too old)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @playcanvas/splat-transform

# For LichtFeld Studio we need GCC/G++ 14
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 60 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 60

WORKDIR /app

# LichtFeld Studio install
RUN git clone https://github.com/microsoft/vcpkg.git
RUN cd vcpkg && ./bootstrap-vcpkg.sh -disableMetrics
RUN git clone https://github.com/MrNeRF/LichtFeld-Studio
RUN cd LichtFeld-Studio && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=/app/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DBUILD_PORTABLE=ON

RUN (cd LichtFeld-Studio && cmake --build build -- -j$(nproc))
ENV LICHTFELD_BIN=/app/LichtFeld-Studio/build/LichtFeld-Studio
ENV LICHTFELD_CONFIG=/app/lichtfeld_optimization_params.json

RUN python3.12 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY lichtfeld_optimization_params.json .

EXPOSE 8080

# Create user and permissions
#RUN groupadd -g 101 compute-node && \
#    useradd -m -u 100 -g compute-node -s /bin/bash compute-node && \
#    chown -R compute-node:compute-node /app

#USER compute-node

CMD ["python", "train_server.py"]
