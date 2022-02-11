# Pointerformer

We provide code, data and trained models. 

## Requirements

```
conda create -n python=3.7
pip install -r requirements.txt
```

## Train

train a model on TSP100.

`python train.py graph_size=100 group_size=100`

## Evaluate

eval on tsp_random

`python eval.py val_data_path=./data/tsp100_test_concorde.txt load_checkpoint_path=./result_ckpt/model100.ckpt`

eval on tsp_partner

`python eval.py val_data_path=./data/partner_100.txt load_checkpoint_path=./result_ckpt/model100.ckpt real_data=true`