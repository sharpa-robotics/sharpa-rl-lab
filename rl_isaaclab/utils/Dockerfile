FROM sharpadev/sharpawavesdk:4.3.2

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    python3-venv \
    libevdev-dev

RUN pip install --upgrade \
    pip \
    setuptools \
    wheel

RUN pip install \
    gymnasium==1.2.1 \
    omegaconf==2.3.0 \
    pynput \
    termcolor \
    tensorboardX

RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0

RUN pip cache purge

RUN apt-get remove -y \
    build-essential \
    libevdev-dev

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
