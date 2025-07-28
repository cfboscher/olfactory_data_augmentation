# 🧪 Olfactory Synthetic Dataset (D2)

This repository contains the **Olfactory Synthetic Dataset (D2)**, created for the study _"Enhancing Low-Resource Sensory Text Classification with LLM-Generated Corpora: A Case Study on Olfactory Reference Extraction"_.

## 📄 Description

**D2** is a synthetic corpus generated and annotated using **GPT-4o**, designed to support research in **olfactory information extraction** under low-resource conditions. It aims to complement real-world data by providing additional, diverse examples of both olfactory and non-olfactory sentences.

The dataset contains:

- **500 positive (olfactory)** sentences
- **1,700 negative (non-olfactory)** sentences
- Two annotation types:
  - `corpus_D2EX.csv`: Manually annotated by a sensory linguistics expert
  - `corpus_D2LM.csv`: Automatically annotated using GPT-4o

Each sentence is annotated at both the **sentence-level** (olfactory vs. not) and **token-level** (identified terms and true positives).

## 📦 Files

```
dataset/
├── corpus_D2EX.csv          # Human-annotated data
├── corpus_D2LM.csv          # LLM-annotated data
├── prompts.txt        # Prompts used for generation and annotation
├── data_dictionary.md # Description of dataset fields
└── README.md          # This file
```

## 🧾 Data Format

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

## 🔍 Use Cases

- **Binary Sentence Classification**: Does a sentence contain olfactory information?
- **Sensory Trms xtraction**: Which specific terms refer to smells?
- **LLM evaluation**: Compare human vs. model annotation consistency

## 🧠 Generation Method

Sentences were generated using GPT-4o with two tailored prompts:
- `P1`: Generate olfactory (positive) sentences
- `P2`: Generate non-olfactory (negative) sentences

Annotation was conducted using:
- `P3`: GPT-4o extraction of sensory terms (`D2_LM.csv`)
- Manual expert annotation for comparison (`D2_EX.csv`)

## 📊 Dataset Statistics

| Type        | Count | Annotation     |
|-------------|-------|----------------|
| Positive    | 500   | Human & LLM    |
| Negative    | 1700  | Human & LLM    |
| Terms       | Token-level (extracted spans) |

## ✅ License

This dataset consists entirely of **synthetic sentences** and annotations over public-domain content. It is released under the **CC-BY 4.0 License**.

## 📚 Citation

If you use this dataset in your research, please cite:

> Anonymized Authors (2025). _Using LLM-generated data for sensory information extraction: a case study on olfactory text._ [PDF](https://anonymous.4open.science/r/ijcnlp_2025-DC51)

## 🔗 Related Resources

- Odeuropa Corpus (D1): https://github.com/Odeuropa/benchmarks_and_corpora  
- Paper and project page: https://anonymous.4open.science/r/ijcnlp_2025-DC51

---

_This dataset supports reproducible research in sensory language modeling and low-resource text classification._
