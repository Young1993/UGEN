# UGEN

```text
Incorporating Instructional Prompts into A Unified Generative Framework for Joint Multiple Intent Detection and Slot Filling
```

```text
UGEN is trained on MixATIS_clean and MixSNIP_clean, respectively.
```

### Code structure

- config：base configuration
- data：dataset
  - MixATIS_clean
  - MixSNIPS_clean
- model: the code of model
- cache：save processed data 
- utils: utils tool
- output：the report
- train.py entry of training
- train.sh shell of training

### Process data

- For MixATIS_clean: 
  - intent/slot: 18/78
  - the max number of intents <= 3 
  - dataset train/dev/test: 13162｜759｜828

- For MixSNIP_clean: 
  - intent/slot 7/39
  - the max number of intents <= 3 
  - dataset train/dev/test：39766/2198/2199
  
### Model
```text
T5-base is used as backbone.
```

### train
- UGEN for full-data

```shell
sh train_qa.sh qa_full
```

### evaluate
```shell
sh train.sh
```

