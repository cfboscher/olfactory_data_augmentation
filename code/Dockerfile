FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Install only essential system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda and clean up
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Create and populate conda env in one layer to reduce image size
RUN conda create -y -n sensory-artificial-data python=3.7 && \
    conda run -n sensory-artificial-data pip install --no-cache-dir --upgrade pip && \
    conda run -n sensory-artificial-data pip install --no-cache-dir \
        tensorflow==2.8.0 \
        torch==1.12.1+cu113 \
        torchvision==0.13.1+cu113 \
        torchaudio==0.12.1 \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    conda run -n sensory-artificial-data pip install --no-cache-dir \
        numpy==1.21.6 \
        pandas==1.3.5 \
        scikit-learn==1.0.2 \
        spacy==3.3.1 \
        transformers==4.20.1 \
        tqdm==4.64.0 \
        pyspellchecker \
        gensim \
        nltk \
        protobuf==3.20\
        ipython==7.33.0 && \
    conda clean -afy && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# Set working directory
WORKDIR /workspace

# Default to bash in the conda env
CMD ["conda", "run", "--no-capture-output", "-n", "sensory-artificial-data", "bash"]