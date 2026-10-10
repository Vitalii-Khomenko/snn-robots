FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        bison \
        build-essential \
        ccache \
        cmake \
        curl \
        dfu-util \
        flex \
        gperf \
        libffi-dev \
        libncurses-dev \
        libssl-dev \
        libusb-1.0-0 \
        git \
        ninja-build \
        unzip \
        wget \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

COPY build.sh /usr/local/bin/stackchan-firmware-build
RUN chmod 0755 /usr/local/bin/stackchan-firmware-build

WORKDIR /project
