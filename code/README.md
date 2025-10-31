# Olfactory Data Augmentation
The code of "Enhancing Low-Resource Sensory Text Classification with LLM-Generated
Corpora : A Case Study on Olfactory Reference Extraction".


## Setup environment with Docker

### Prerequisites

- Docker installed on your system  
- NVIDIA Container Toolkit installed (for GPU support)  
- Optional: Docker Compose (if using a `docker-compose.yml` file)

---

### 1. Build the Docker Image

Open a terminal, navigate to this directory containing the `Dockerfile`, and run:

```bash
docker build -t sensory-artificial-data:latest .
``` 
This will build a Docker image named ```sensory-artificial-data``` using the provided ```Dockerfile```.


### 2. Run the Docker Container

To run the container with GPU support and an interactive shell:

```bash
docker run --gpus all -it  -v .:/workspace   --name sensory-artificial-data sensory-artificial-data:latest
``` 

Start and attach the container :     


```bash
 docker start sensory-artificial-data
 docker attach sensory-artificial-data
``` 

Inside the container, get into the conda environment and set the PYTHONPATH : 
```bash
conda activate sensory-artificial-data && export PYTHONPATH=$PWD/src/
```

### 3. Run Experiments 

Run the experiment of your choice using python such that : 

```bash
python <script_name>
```



## Code Artifacts

This repository includes code artifacts from the following external projects : 

- BERT for sentiment classification using PyTorch : https://github.com/Taaniya/bert-for-sentiment-classification-pytorch
- Extracting Phrase from Sentence : https://github.com/Jitendra-Dash/Extracting-Phrase-From-Sentence
- Sensorimotor Distance Calculator : https://github.com/emcoglab/sensorimotor-distance-calculator
- SENSE-LM : https://github.com/cfboscher/sense-lm
- BERT : https://huggingface.co/bert-base-uncased
- MacBERTh : https://huggingface.co/emanjavacas/MacBERTh
- RoBERTa : https://huggingface.co/roberta-base


