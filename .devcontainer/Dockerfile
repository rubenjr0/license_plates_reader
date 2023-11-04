FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.11
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && export DEBIAN_FRONTEND=noninteractive \
    && apt upgrade -y \
    && apt install ffmpeg libopencv-dev tesseract-ocr -y \
    && apt autoremove -y && apt autoclean -y

ARG USER=vscode
USER $USER
ARG HOME="/home/$USER"

ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${HOME}/.local/bin:$PATH"

ARG PYTHON_VERSION=3.11
ARG POETRY_VERSION=1.6.1

RUN echo "Installing PyEnv" \
    && curl https://pyenv.run | bash 

RUN echo "Installing Python $PYTHON_VERSION" \
    && pyenv install ${PYTHON_VERSION}

RUN echo "Setting Python $PYTHON_VERSION as global" \
    && pyenv global ${PYTHON_VERSION}

RUN echo "Installing Poetry" \
    && curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}

RUN echo "Configuring Poetry" \
    && poetry config virtualenvs.in-project false