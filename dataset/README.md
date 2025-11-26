# Olfactory Synthetic Dataset

This repository contains the **Olfactory Synthetic Dataset **, created for the study _"Enhancing Low-Resource Sensory Text Classification with LLM-Generated Corpora: A Case Study on Olfactory Reference Extraction"_ and referred to as D2 in the latter.

## Description

**D2** is a synthetic corpus generated and annotated using **GPT-4o**, designed to support research in **olfactory information extraction** under low-resource conditions. It aims to complement real-world data by providing additional, diverse examples of both olfactory and non-olfactory sentences.

The dataset contains:

- **500 positive (olfactory)** sentences
- **1,700 negative (non-olfactory)** sentences
- Two annotation types:
  - `corpus_D2EX.csv`: Manually annotated by a human expert
  - `corpus_D2LM.csv`: Automatically annotated using GPT-4o

Each sentence is annotated at both the **sentence-level** (olfactory vs. not) and **token-level** (identified terms and true positives).

##  Files

```
dataset/
├── corpus_D2EX.csv          # Human-annotated data
├── corpus_D2LM.csv          # LLM-annotated data
└── README.md          # This file
```

## Data Format

Each `.csv` file contains the following columns:

- `phrases`: The synthetic sentence
- `identified_terms`: Terms identified as potentially olfactory
- `positive_negative`: `1` if the sentence is olfactory, `0` if not

Example row:
```
phrases: "The aroma of freshly baked bread filled the room."
identified_terms: ['aroma', 'bread']
positive_negative:             1
```

## Dataset Generation and Annotation Prompts

We provide the prompts used for the generation and automatic annotation of the **$D_2$ dataset** in the table below. The **P1** prompt is used to generate **positive examples**, and conversely, **P2** is used to generate **negative examples**. Considering the limitations of the GPT-4o web application, we generate the dataset in batches of 100 examples, which are compiled into a CSV file along with their class at a sentence level (positive/negative).  

Then, the prompt **P3** is applied on positive examples to extract **positive terms** as part of the **D2_LM annotation**.


Annotation was conducted using:
- Manual expert annotation for comparison (`corpus_D2EX.csv`)
- `P3`: GPT-4o extraction of sensory terms (`corpus_D2LM.csv`)


| **Type**                    | **Prompt Description** |
|----------------------------|------------------------|
| **P1 (Positive)**          | *"Could you generate 100 sentences of 10 words each, containing references to olfactory experiences, and avoid repeating the same sentence structures? You may include different kinds of descriptions: what produces the olfactory experience or the quality of smell, for different types of scents (people, objects, or environment)."* |
| **P2 (Negative)**          | *"Could you generate 100 sentences with 10 words for each, making sure they absolutely do not make any reference to any olfactory experience, and avoid repeating the same sentence structures?"* |
| **P3 (Positive Terms Annotation)** | *"Extract words from the following sentences that evoke smells, explicitly or implicitly (e.g., describing smell quality or source). For example, from ‘Musk pots generally moist exhales disagreeable predominant ammoniacal smell...’ extract ‘disagreeable, predominant, ammoniacal, musk, smell.’"* |



## Related Resources

- Odeuropa Corpus (D1): https://github.com/Odeuropa/benchmarks_and_corpora  
- Paper and project page : Released upon article publication at IJCNLP-AACL 2025

