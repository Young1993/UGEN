# UGEN

```text
Incorporating Instructional Prompts into A Unified Generative Framework for Joint Multiple Intent Detection and Slot Filling

@inproceedings{wu-etal-2022-incorporating,
    title = "Incorporating Instructional Prompts into a Unified Generative Framework for Joint Multiple Intent Detection and Slot Filling",
    author = "Wu, Yangjun  and
      Wang, Han  and
      Zhang, Dongxiang  and
      Chen, Gang  and
      Zhang, Hao",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.631",
    pages = "7203--7208",
    abstract = "The joint multiple Intent Detection (ID) and Slot Filling (SF) is a significant challenge in spoken language understanding. Because the slots in an utterance may relate to multi-intents, most existing approaches focus on utilizing task-specific components to capture the relations between intents and slots. The customized networks restrict models from modeling commonalities between tasks and generalization for broader applications. To address the above issue, we propose a Unified Generative framework (UGEN) based on a prompt-based paradigm, and formulate the task as a question-answering problem. Specifically, we design 5-type templates as instructional prompts, and each template includes a question that acts as the driver to teach UGEN to grasp the paradigm, options that list the candidate intents or slots to reduce the answer search space, and the context denotes original utterance. Through the instructional prompts, UGEN is guided to understand intents, slots, and their implicit correlations. On two popular multi-intent benchmark datasets, experimental results demonstrate that UGEN achieves new SOTA performances on full-data and surpasses the baselines by a large margin on 5-shot (28.1{\%}) and 10-shot (23{\%}) scenarios, which verify that UGEN is robust and effective.",
}
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

### install
```shell
pip install -r requirements.txt
```

### train
- Example on MixSNIP
  UGEN for full-data 

```shell
sh train_qa.sh qa_full MixSNIP MixSNIP_clean
```

- Example on MixATIS
  UGEN for full-data 
```shell
sh train_qa.sh qa_full MixATIS MixATIS_clean
```

## Contact
If you have any question, please email at yjwu@zjuici.com

