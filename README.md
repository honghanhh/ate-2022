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
|Domains|Corruption,  Wind energy, Equitation, Heart failure|Biomechanics, Chemistry, Veterinary, Linguistics |

## 4. Implementation

As the orginal dataset does not follow IOB format, we preprocess the data to sequentially map each token with it regarding label. An example of IOB format is demontrated below.

![](./imgs/ex.png)

For ACTER dataset, run the following command to preprocess the data:

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

### 5.1 ACTER dataset
We report the following results from all scenarios in terms of $F_{1}$ score for ACTER dataset. The full results are available later in our paper. Note that we train on Corruptions, Wind energy (with/without domains Slovenian), validate on Equitation, and test on Heart failure domains (tri = en + fr + nl).

<center>
<table>
<tr><th>Test = en </th><th>Test = fr</th><th>Test =nl</th></tr></tr>
<tr><td>

|Train          |ANN    |NES    |	
|:-----:        |:-----:|:-----:|
|en	            |52.63  | 56.61	|
|fr	            |41.95  | 47.33	|
|nl	            |56.00  | 57.97	|
|tri	        |54.86 | 56.35	|
|tri + sl       |54.23  | 55.63	|

</td><td>

|Train          |ANN    |NES    |	
|:-----:        |:-----:|:-----:|
|en	            |55.75  | 61.07	|
|fr	            |54.51  | 58.11	|
|nl	            |58.03  | 59.52	|
|tri            |58.07 | 59.78	|
|tri + sl       |59.81  | 60.96	|

</td><td>

|Train          |ANN    |NES    |	
|:-----:        |:-----:|:-----:|
|en	            |64.91  | 67.63	|
|fr	            |59.76  | 63.29	|
|nl	            |65.95  | 66.87	|
|tri            |67.00  | 67.86	|
|tri + sl|68.54  | 68.26	|

</td></tr> </table>
</center>

### 5.2 RSDO5 dataset

We report the following results from all scenarios in terms of $F_{1}$ score for RSDO5 dataset. The full results are available later in our paper.

#### 5.2.1 XLM-RoBERTa in mono-lingual settings

<center>
<table>
<tr><th>Test = ling </th><th>Test = vet</th></tr></tr>
<tr><td>

|Validation     |Test   | F1-score    |
| :-: | :-: | :-: |
|vet    |ling   | 66.69  | 
|bim    |ling   | 71.51  | 
|kem    |ling   | 69.15  |

</td><td>

|Validation     |Test   | F1-score    |
| :-: | :-: | :-: | 
|ling   |vet    | 68.82 | 
|kem    |vet    | 68.94  | 
|bim    |vet    | 68.68  |

</td><td></table>
</center>

<center>
<table>
<tr><th>Test = kem </th><th>Test = bim</th></tr></tr>
<tr><td>

|Validation     |Test   | F1-score    |
| :-: | :-: | :-: | 
|ling   |kem    |61.16 |
|bim    |kem    |64.83  |
|vet    |kem    |64.27  | 

</td><td>

|Validation     |Test   | F1-score    |
| :-: | :-: | :-: | 
|vet    |bim    |65.11  |
|ling   |bim    |63.69  | 
|kem    |bim    |63.16  |

</td><td></table>
</center>

#### 5.2.1 XLM-RoBERTa in multi-lingual settings

<center>
<table>
<tr><th>Test = ling </th><th>Test = vet</th></tr></tr>
<tr><td>

|Validation     |Test   |ANN    |NES    |
| :-: | :-: | :-: | :-: |
|vet    |ling   |68.60  | 68.51	|
|bim    |ling   |67.92  | 68.17	|
|kem    |ling   |68.84  | 68.46 |

</td><td>

|Validation     |Test   |ANN    |NES    |
| :-: | :-: | :-: | :-: |
|ling   |vet    | 68.00 | 68.30	|
|kem    |vet    |69.29  | 69.09	|
|bim    |vet    |69.09  | 66.91	|

</td><td></table>
</center>

<center>
<table>
<tr><th>Test = kem </th><th>Test = bim</th></tr></tr>
<tr><td>

|Validation     |Test   |ANN    |NES    |
| :-: | :-: | :-: | :-: |
|ling   |kem    |63.45  | 60.38	|
|bim    |kem    |65.14  | 59.86	|
|vet    |kem    |63.64  | 63.28	|

</td><td>

|Validation     |Test   |ANN    |NES    |
| :-: | :-: | :-: | :-: |
|vet    |bim    |62.98  | 63.68	|
|ling   |bim    |62.13  | 62.44	|
|kem    |bim    |62.26  | 64.31	|

</td><td></table>
</center>

## References

Lang, C., Wachowiak, L., Heinisch, B., & Gromann, D. Transforming Term Extraction: Transformer-Based Approaches to Multilingual Term Extraction Across Domains. [(PDF)](https://aclanthology.org/2021.findings-acl.316.pdf)

## Contributors:
- 🐮 [TRAN Thi Hong Hanh](https://github.com/honghanhh) 🐮
- Prof. [Senja POLLAK](https://github.com/senjapollak)
- Prof. [Antoine DOUCET](https://github.com/antoinedoucet)
- Prof. [Matej MARTINC](https://github.com/matejMartinc)
