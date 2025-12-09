FROM ghcr.io/nerfstudio-project/nerfstudio:latest

ARG USERNAME=splatter-server
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG DEBIAN_FRONTEND=noninteractive

# Install Packages
RUN python3 -m pip install ply2splat

#Copy Files
RUN mkdir /app

WORKDIR /app

COPY ./*py /app

# Run reconstruction server as separate user, not root
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME