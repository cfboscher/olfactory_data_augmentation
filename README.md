# Enhancing Low-Resource Sensory Text Classification with LLM-Generated Corpora:  A Case Study on Olfactory Reference Extraction

## Description

The repository of ["Enhancing Low-Resource Sensory Text Classification with LLM-Generated
Corpora : A Case Study on Olfactory Reference Extraction"](https://openreview.net/forum?id=MVlikveDSz) (Accepted in the Proceedings of [IJCNLP-AACL 2025](https://www.afnlp.org/conferences/ijcnlp2025/) -- Full citation upon publication release) 

## Usage
- The Olfactory Artificial Dataset can be found in the [```dataset```](https://github.com/cfboscher/olfactory_data_augmentation/tree/main/dataset) directory. For detailed description of the dataset, please see ```dataset/README.md```.

- The source code of experiments along with instructions to run the code and reproduct experiments are in [```code```](https://github.com/cfboscher/olfactory_data_augmentation/tree/main/code). Please refer to [```code/README.md```](https://github.com/cfboscher/olfactory_data_augmentation/tree/main/code/README.md) for further instructions.



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
