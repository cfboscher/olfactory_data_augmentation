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


## Citation

Please cite the original paper in case of use of code or/and dataset. 

>Cédric Boscher, Shannon Bruderer, Christine Largeron, Véronique Eglin, and Elöd Egyed-Zsigmond. 2025. [Enhancing Low-Resource Text Classification with LLM-Generated Corpora : A Case Study on Olfactory Reference Extraction](https://aclanthology.org/2025.ijcnlp-long.161/). *In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics*, pages 3004–3027, Mumbai, India. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics.

```
@inproceedings{boscher-etal-2025-enhancing,
    title = "Enhancing Low-Resource Text Classification with {LLM}-Generated Corpora : A Case Study on Olfactory Reference Extraction",
    author = {Boscher, C{\'e}dric  and
      Bruderer, Shannon  and
      Largeron, Christine  and
      Eglin, V{\'e}ronique  and
      Egyed-Zsigmond, El{\"o}d},
    editor = "Inui, Kentaro  and
      Sakti, Sakriani  and
      Wang, Haofen  and
      Wong, Derek F.  and
      Bhattacharyya, Pushpak  and
      Banerjee, Biplab  and
      Ekbal, Asif  and
      Chakraborty, Tanmoy  and
      Singh, Dhirendra Pratap",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.161/",
    pages = "3004--3027",
    ISBN = "979-8-89176-298-5",
    abstract = "Extracting sensory information from text, particularly olfactory references, is challenging due to limited annotated datasets and the implicit, subjective nature of sensory experiences. This study investigates whether GPT-4o-generated data can complement or replace human annotations. We evaluate human- and LLM-labeled corpora on two tasks: coarse-grained detection of olfactory content and fine-grained sensory term extraction. Despite lexical variation, generated texts align well with real data in semantic and sensorimotor embedding spaces. Models trained on synthetic data perform strongly, especially in low-resource settings. Human annotations offer better recall by capturing implicit and diverse aspects of sensoriality, while GPT-4o annotations show higher precision through clearer pattern alignment. Data augmentation experiments confirm the utility of synthetic data, though trade-offs remain between label consistency and lexical diversity. These findings support using synthetic data to enhance sensory information mining when annotated data is limited."
}
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


