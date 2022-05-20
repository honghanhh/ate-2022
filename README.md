# Comparison among Mono-lingual, Cross-lingual, and Multi-lingual approaches to Automatic Term Extraction

## 1. Description

In this repo, we experiment  XLM-RoBERTa on ACTER and RSDO5 dataset to evaluate the abilities of (zero-shot) cross-lingual and multi-lingual learning in comparison with the mono-lingual setting in the cross-domain sequence labeling automatic term extraction (ATE) task and compare our models' performance towards the benchmarks.

---

## 2. Requirements

Please install all the necessary libraries noted in [requirements.txt](./requirements.txt) using this command:

```
pip install -r requirements.txt
```

## 3. Data

The experiments were conducted on 2 datasets:

||ACTER dataset| RSDO5 dataset|
|:-:|:-:|:-:|
|Languages|English, French, and Dutch|Slovenian|
|Domains|Corruption,  Wind energy, Equitation, Heart failure|Biomechanics, Chemistry, Veterinary, and Linguistics|

## 4. Implementation

As the orginal dataset does not follow IOB format, we preprocess the data to sequentially map each token with it regarding label. For ACTER dataset, run the following command to preprocess the data:

```python
preprocess.py [-corpus_path] [-term_path] [-output_csv_path] [-language]
```

where `-corpus_path` is the path to the directory containing the corpus files, `-term_path` is the path to the directory containing the term files, `-output_csv_path` is the path to the output csv file, and `-language` is the language of the corpus.

For RSDO5 dataset, the dataset is already in conll format. Please use `read_conll()` function in `sl_preprocess.py` to get the mapping.

Run the following command to train the model with all the scenarios in ACTER and RSDSO5 datasets:

```python
run.sh
```

where `run.sh` covers the following scenarios:

- ACTER dataset with XLM-RoBERTa in mono-lingual, cros-lingual, and multi-lingual settings with both ANN and NES version with multi-lingual settings covering only three languages from ACTER and additional Slovenian add-ons (10 scenarios).

- RSDO5 dataset with XLM-RoBERTa in mono-lingual, cros-lingual, and multi-lingual settings with cross-lingual and multi-lingual taking into account the ANN and NES version (48 scenarios).


## 5. Results



## References

Lang, C., Wachowiak, L., Heinisch, B., & Gromann, D. Transforming Term Extraction: Transformer-Based Approaches to Multilingual Term Extraction Across Domains. [(PDF)](https://aclanthology.org/2021.findings-acl.316.pdf)

## Contributors:
- 🐮 [TRAN Thi Hong Hanh](https://github.com/honghanhh) 🐮
- Prof. [Senja POLLAK](https://github.com/senjapollak)
- Prof. [Antoine DOUCET](https://github.com/antoinedoucet)
- Prof. [Matej MARTINC](https://github.com/matejMartinc)
