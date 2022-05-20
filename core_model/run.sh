# 1. ACTER 

## 1.1 ANN version
### 1.1.1 Mono- and cross-lingual
python acter_cross.py -ver ann -store ./ann_res1/ -preds ./results/acter/ann_en.txt  -log ./results/acter/ann_en_log.txt -lang en
python acter_cross.py -ver ann -store ./ann_res2/ -preds ./results/acter/ann_fr.txt  -log ./results/acter/ann_fr_log.txt -lang fr
python acter_cross.py -ver ann -store ./ann_res3/ -preds ./results/acter/ann_nl.txt  -log ./results/acter/ann_nl_log.txt -lang du

### 1.1.2 Multi-lingual
python acter_multi.py -ver ann -store ./ann_res4/ -preds ./results/acter/ann_tri.txt  -log ./results/acter/ann_tri_log.txt -lang tri
python acter_multi.py -ver ann -store ./ann_res5/ -preds ./results/acter/ann_slo.txt  -log ./results/acter/ann_slo_log.txt -lang slo

# 1.2 NES version
### 1.2.1 Mono- and cross-lingual
python acter_cross.py -ver nes -store ./nes_res1/ -preds ./results/acter/nes_en.txt  -log ./results/acter/nes_en_log.txt -lang en
python acter_cross.py -ver nes -store ./nes_res2/ -preds ./results/acter/nes_fr.txt  -log ./results/acter/nes_fr_log.txt -lang fr
python acter_cross.py -ver nes -store ./nes_res3/ -preds ./results/acter/nes_nl.txt  -log ./results/acter/nes_nl_log.txt -lang du

### 1.2.2 Multi-lingual
python acter_multi.py -ver nes -store ./nes_res4/ -preds ./results/acter/nes_tri.txt  -log ./results/acter/nes_tri_log.txt -lang tri
python acter_multi.py -ver nes -store ./nes_res5/ -preds ./results/acter/nes_slo.txt  -log ./results/acter/nes_slo_log.txt -lang slo

# 2. Slovenian dataset

## 2.1. Mono-lignual

# ___Ling___
python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/bim.csv -val ./processed_data/new_sl/vet.csv -test ./processed_data/new_sl/jez.csv -gold_val ./ACTER/sl/termlists_2/rsdo5vet.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5jez.terms2 -store ./slo_res1/ -preds ./results/multi_xlm/nes/vet_ling.txt  -log ./results/multi_xlm/nes/vet_ling_log.txt -multi False

python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/bim.csv -test ./processed_data/new_sl/jez.csv -gold_val ./ACTER/sl/termlists_2/rsdo5bim.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5jez.terms2 -store ./slo_res2/ -preds ./results/multi_xlm/nes/bim_ling.txt  -log ./results/multi_xlm/nes/bim_ling_log.txt -multi False

python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/kem.csv -test ./processed_data/new_sl/jez.csv -gold_val ./ACTER/sl/termlists_2/rsdo5kem.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5jez.terms2 -store ./slo_res3/ -preds ./results/multi_xlm/nes/kem_ling.txt  -log ./results/multi_xlm/nes/kem_ling_log.txt -multi False

# ___Vet___
python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/kem.csv -val ./processed_data/new_sl/jez.csv -test ./processed_data/new_sl/vet.csv -gold_val ./ACTER/sl/termlists_2/rsdo5jez.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5vet.terms2 -store ./slo_res4/ -preds ./results/multi_xlm/nes/ling_vet.txt  -log ./results/multi_xlm/nes/ling_vet_log.txt -multi False

python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/kem.csv -test ./processed_data/new_sl/vet.csv -gold_val ./ACTER/sl/termlists_2/rsdo5kem.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5vet.terms2 -store ./slo_res5/ -preds ./results/multi_xlm/nes/kem_vet.txt  -log ./results/multi_xlm/nes/kem_vet_log.txt -multi False

python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/bim.csv -test ./processed_data/new_sl/vet.csv -gold_val ./ACTER/sl/termlists_2/rsdo5bim.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5vet.terms2 -store ./slo_res6/ -preds ./results/multi_xlm/nes/bim_vet.txt  -log ./results/multi_xlm/nes/bim_vet_log.txt -multi False

# ___Kem___
python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/jez.csv -test ./processed_data/new_sl/kem.csv -gold_val ./ACTER/sl/termlists_2/rsdo5jez.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5kem.terms2 -store ./slo_res7/ -preds ./results/multi_xlm/nes/ling_kem.txt  -log ./results/multi_xlm/nes/ling_kem_log.txt -multi False

python slo_multi.py -train1 ./processed_data/new_sl/vet.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/bim.csv -test ./processed_data/new_sl/kem.csv -gold_val ./ACTER/sl/termlists_2/rsdo5bim.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5kem.terms2 -store ./slo_res8/ -preds ./results/multi_xlm/nes/bim_kem.txt  -log ./results/multi_xlm/nes/bim_kem_log.txt -multi False

python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/vet.csv -test ./processed_data/new_sl/kem.csv -gold_val ./ACTER/sl/termlists_2/rsdo5vet.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5kem.terms2 -store ./slo_res9/ -preds ./results/multi_xlm/nes/vet_kem.txt  -log ./results/multi_xlm/nes/vet_kem_log.txt -multi False

# ___Bim___
python slo_multi.py -train1 ./processed_data/new_sl/jez.csv -train2 ./processed_data/new_sl/kem.csv -val ./processed_data/new_sl/vet.csv -test ./processed_data/new_sl/bim.csv -gold_val ./ACTER/sl/termlists_2/rsdo5vet.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5bim.terms2 -store ./slo_res10/ -preds ./results/multi_xlm/nes/vet_bim.txt  -log ./results/multi_xlm/nes/vet_bim_log.txt -multi False

python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/jez.csv -test ./processed_data/new_sl/bim.csv -gold_val ./ACTER/sl/termlists_2/rsdo5jez.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5bim.terms2 -store ./slo_res11/ -preds ./results/multi_xlm/nes/ling_bim.txt  -log ./results/multi_xlm/nes/ling_bim_log.txt -multi False

python slo_multi.py -train1 ./processed_data/new_sl/vet.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/kem.csv -test ./processed_data/new_sl/bim.csv -gold_val ./ACTER/sl/termlists_2/rsdo5kem.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5bim.terms2 -store ./slo_res12/ -preds ./results/multi_xlm/nes/kem_bim.txt  -log ./results/multi_xlm/nes/kem_bim_log.txt -multi False


## 2.2. Cross-lingual
### 2.2.1. ANN version
### 2.2.2. NES version


## 2.3. Multi-lingual
### 2.3.1. ANN version

# ___Ling___
python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/bim.csv -val ./processed_data/new_sl/vet.csv -test ./processed_data/new_sl/jez.csv -gold_val ./ACTER/sl/termlists_2/rsdo5vet.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5jez.terms2 -store ./slo_ann_res1/ -preds ./results/multi_xlm/nes/vet_ling.txt  -log ./results/multi_xlm/nes/vet_ling_log.txt -ver ann

python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/bim.csv -test ./processed_data/new_sl/jez.csv -gold_val ./ACTER/sl/termlists_2/rsdo5bim.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5jez.terms2 -store ./slo_ann_res2/ -preds ./results/multi_xlm/nes/bim_ling.txt  -log ./results/multi_xlm/nes/bim_ling_log.txt -ver ann

python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/kem.csv -test ./processed_data/new_sl/jez.csv -gold_val ./ACTER/sl/termlists_2/rsdo5kem.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5jez.terms2 -store ./slo_ann_res3/ -preds ./results/multi_xlm/nes/kem_ling.txt  -log ./results/multi_xlm/nes/kem_ling_log.txt -ver ann

# ___Vet___
python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/kem.csv -val ./processed_data/new_sl/jez.csv -test ./processed_data/new_sl/vet.csv -gold_val ./ACTER/sl/termlists_2/rsdo5jez.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5vet.terms2 -store ./slo_ann_res4/ -preds ./results/multi_xlm/nes/ling_vet.txt  -log ./results/multi_xlm/nes/ling_vet_log.txt -ver ann

python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/kem.csv -test ./processed_data/new_sl/vet.csv -gold_val ./ACTER/sl/termlists_2/rsdo5kem.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5vet.terms2 -store ./slo_ann_res5/ -preds ./results/multi_xlm/nes/kem_vet.txt  -log ./results/multi_xlm/nes/kem_vet_log.txt -ver ann

python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/bim.csv -test ./processed_data/new_sl/vet.csv -gold_val ./ACTER/sl/termlists_2/rsdo5bim.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5vet.terms2 -store ./slo_ann_res6/ -preds ./results/multi_xlm/nes/bim_vet.txt  -log ./results/multi_xlm/nes/bim_vet_log.txt -ver ann

# ___Kem___
python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/jez.csv -test ./processed_data/new_sl/kem.csv -gold_val ./ACTER/sl/termlists_2/rsdo5jez.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5kem.terms2 -store ./slo_ann_res7/ -preds ./results/multi_xlm/nes/ling_kem.txt  -log ./results/multi_xlm/nes/ling_kem_log.txt -ver ann

python slo_multi.py -train1 ./processed_data/new_sl/vet.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/bim.csv -test ./processed_data/new_sl/kem.csv -gold_val ./ACTER/sl/termlists_2/rsdo5bim.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5kem.terms2 -store ./slo_ann_res8/ -preds ./results/multi_xlm/nes/bim_kem.txt  -log ./results/multi_xlm/nes/bim_kem_log.txt -ver ann

python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/vet.csv -test ./processed_data/new_sl/kem.csv -gold_val ./ACTER/sl/termlists_2/rsdo5vet.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5kem.terms2 -store ./slo_ann_res9/ -preds ./results/multi_xlm/nes/vet_kem.txt  -log ./results/multi_xlm/nes/vet_kem_log.txt -ver ann

# ___Bim___
python slo_multi.py -train1 ./processed_data/new_sl/jez.csv -train2 ./processed_data/new_sl/kem.csv -val ./processed_data/new_sl/vet.csv -test ./processed_data/new_sl/bim.csv -gold_val ./ACTER/sl/termlists_2/rsdo5vet.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5bim.terms2 -store ./slo_ann_res10/ -preds ./results/multi_xlm/nes/vet_bim.txt  -log ./results/multi_xlm/nes/vet_bim_log.txt -ver ann

python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/jez.csv -test ./processed_data/new_sl/bim.csv -gold_val ./ACTER/sl/termlists_2/rsdo5jez.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5bim.terms2 -store ./slo_ann_res11/ -preds ./results/multi_xlm/nes/ling_bim.txt  -log ./results/multi_xlm/nes/ling_bim_log.txt -ver ann

python slo_multi.py -train1 ./processed_data/new_sl/vet.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/kem.csv -test ./processed_data/new_sl/bim.csv -gold_val ./ACTER/sl/termlists_2/rsdo5kem.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5bim.terms2 -store ./slo_ann_res12/ -preds ./results/multi_xlm/nes/kem_bim.txt  -log ./results/multi_xlm/nes/kem_bim_log.txt -ver ann

### 2.3.3. NES version

# ___Ling___
python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/bim.csv -val ./processed_data/new_sl/vet.csv -test ./processed_data/new_sl/jez.csv -gold_val ./ACTER/sl/termlists_2/rsdo5vet.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5jez.terms2 -store ./slo_nes_res1/ -preds ./results/multi_xlm/nes/vet_ling.txt  -log ./results/multi_xlm/nes/vet_ling_log.txt -ver nes

python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/bim.csv -test ./processed_data/new_sl/jez.csv -gold_val ./ACTER/sl/termlists_2/rsdo5bim.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5jez.terms2 -store ./slo_nes_res2/ -preds ./results/multi_xlm/nes/bim_ling.txt  -log ./results/multi_xlm/nes/bim_ling_log.txt -ver nes

python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/kem.csv -test ./processed_data/new_sl/jez.csv -gold_val ./ACTER/sl/termlists_2/rsdo5kem.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5jez.terms2 -store ./slo_nes_res3/ -preds ./results/multi_xlm/nes/kem_ling.txt  -log ./results/multi_xlm/nes/kem_ling_log.txt -ver nes

# ___Vet___
python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/kem.csv -val ./processed_data/new_sl/jez.csv -test ./processed_data/new_sl/vet.csv -gold_val ./ACTER/sl/termlists_2/rsdo5jez.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5vet.terms2 -store ./slo_nes_res4/ -preds ./results/multi_xlm/nes/ling_vet.txt  -log ./results/multi_xlm/nes/ling_vet_log.txt -ver nes

python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/kem.csv -test ./processed_data/new_sl/vet.csv -gold_val ./ACTER/sl/termlists_2/rsdo5kem.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5vet.terms2 -store ./slo_nes_res5/ -preds ./results/multi_xlm/nes/kem_vet.txt  -log ./results/multi_xlm/nes/kem_vet_log.txt -ver nes

python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/bim.csv -test ./processed_data/new_sl/vet.csv -gold_val ./ACTER/sl/termlists_2/rsdo5bim.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5vet.terms2 -store ./slo_nes_res6/ -preds ./results/multi_xlm/nes/bim_vet.txt  -log ./results/multi_xlm/nes/bim_vet_log.txt -ver nes

# ___Kem___
python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/jez.csv -test ./processed_data/new_sl/kem.csv -gold_val ./ACTER/sl/termlists_2/rsdo5jez.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5kem.terms2 -store ./slo_nes_res7/ -preds ./results/multi_xlm/nes/ling_kem.txt  -log ./results/multi_xlm/nes/ling_kem_log.txt -ver nes

python slo_multi.py -train1 ./processed_data/new_sl/vet.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/bim.csv -test ./processed_data/new_sl/kem.csv -gold_val ./ACTER/sl/termlists_2/rsdo5bim.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5kem.terms2 -store ./slo_nes_res8/ -preds ./results/multi_xlm/nes/bim_kem.txt  -log ./results/multi_xlm/nes/bim_kem_log.txt -ver nes

python slo_multi.py -train1 ./processed_data/new_sl/bim.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/vet.csv -test ./processed_data/new_sl/kem.csv -gold_val ./ACTER/sl/termlists_2/rsdo5vet.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5kem.terms2 -store ./slo_nes_res9/ -preds ./results/multi_xlm/nes/vet_kem.txt  -log ./results/multi_xlm/nes/vet_kem_log.txt -ver nes

# ___Bim___
python slo_multi.py -train1 ./processed_data/new_sl/jez.csv -train2 ./processed_data/new_sl/kem.csv -val ./processed_data/new_sl/vet.csv -test ./processed_data/new_sl/bim.csv -gold_val ./ACTER/sl/termlists_2/rsdo5vet.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5bim.terms2 -store ./slo_nes_res10/ -preds ./results/multi_xlm/nes/vet_bim.txt  -log ./results/multi_xlm/nes/vet_bim_log.txt -ver nes

python slo_multi.py -train1 ./processed_data/new_sl/kem.csv -train2 ./processed_data/new_sl/vet.csv -val ./processed_data/new_sl/jez.csv -test ./processed_data/new_sl/bim.csv -gold_val ./ACTER/sl/termlists_2/rsdo5jez.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5bim.terms2 -store ./slo_nes_res11/ -preds ./results/multi_xlm/nes/ling_bim.txt  -log ./results/multi_xlm/nes/ling_bim_log.txt -ver nes

python slo_multi.py -train1 ./processed_data/new_sl/vet.csv -train2 ./processed_data/new_sl/jez.csv -val ./processed_data/new_sl/kem.csv -test ./processed_data/new_sl/bim.csv -gold_val ./ACTER/sl/termlists_2/rsdo5kem.terms2 -gold_test ./ACTER/sl/termlists_2/rsdo5bim.terms2 -store ./slo_nes_res12/ -preds ./results/multi_xlm/nes/kem_bim.txt  -log ./results/multi_xlm/nes/kem_bim_log.txt -ver nes
