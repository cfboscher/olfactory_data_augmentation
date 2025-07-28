# 🧪 Olfactory Synthetic Dataset (D2)

This repository contains the **Olfactory Synthetic Dataset (D2)**, developed as part of the study _"Enhancing Low-Resource Sensory Text Classification with LLM-Generated Corpora: A Case Study on Olfactory Reference Extraction"_.

## 📄 Description

D2 is a synthetic dataset generated using **GPT-4o** to simulate olfactory (smell-related) and non-olfactory sentences, serving as a complementary resource to the real-world **Odeuropa Corpus (D1)**. The dataset is intended for research on **sensory information extraction**, particularly in **low-resource settings**.

The dataset includes:

- **500 positive (olfactory)** sentences
- **1,700 negative (non-olfactory)** sentences
- Two annotation versions:
  - `D2_EX`: Human-annotated (expert)
  - `D2_LM`: Automatically annotated using GPT-4o

Each positive sentence includes **token-level annotations** for olfactory terms.

## 📦 Structure

```
dataset/
├── D2_EX.csv          # Human-annotated data
├── D2_LM.csv          # LLM-annotated data
├── prompts.txt        # Prompts used for generation
├── data_dictionary.md # Description of dataset fields
└── README.md          # This file
```

## 🔍 Use Cases

- Sentence classification: Does a sentence contain olfactory references?
- Sensory term extraction: Which words refer to olfactory concepts?

## 🧠 Generation Method

- Sentences were created using **two custom prompts**:
  - `P1` for positive (olfactory) examples
  - `P2` for negative (non-olfactory) examples
- Olfactory terms were annotated:
  - Automatically using `P3` (LLM extraction)
  - Manually by a domain expert in sensory linguistics

## 🧪 Sample

| Type     | Sentence                                                            |
|----------|---------------------------------------------------------------------|
| Positive | _“The aroma of **fresh-baked bread** lingered warmly.”_            |
| Negative | _“A fearless diver plumbed unexplored reefs below.”_               |

## 📊 Dataset Statistics

| Type        | Count | Annotation     |
|-------------|-------|----------------|
| Positive    | 500   | Human & LLM    |
| Negative    | 1700  | Human & LLM    |
| Vocabulary  | 902 unique terms (DEX), 318 (DLM) |


## ✅ License

This dataset contains only **synthetically generated text** and **annotations of public-domain content**, and is released under the **CC-BY 4.0 License**.

## 📚 Citation

If you use this dataset in your research, please cite:

> Anonymized Authors (2025). _Using LLM-generated data for sensory information extraction: a case study on olfactory text._ [PDF](https://anonymous.4open.science/r/ijcnlp_2025-DC51)

## 🔗 Related Resources

- Odeuropa Corpus (D1): https://github.com/Odeuropa/benchmarks_and_corpora  
- Paper and project page: https://anonymous.4open.science/r/ijcnlp_2025-DC51

---

_This dataset provides a benchmark for evaluating LLM-generated data in subjective domains like sensory perception, and is part of broader efforts to support reproducible research in low-resource NLP._
