# CoDa : Constrained Generation based Data Augmentation for Low-Resource NLP

Implementation of [CoDa : Constrained Generation based Data Augmentation for Low-Resource NLP](https://arxiv.org/pdf/2404.00415)

![Proposed Methodology](./diagram.png)

### Constraint Generation:

1. For Classification tasks:

```shell
sh classification_pipeline <dataset_name> <dataset_split> <debug_mode> <dataset_split_for_shortcut_grammar>
```

2. For NER tasks:

```shell
sh ner_pipeline <dataset_name> <dataset_split> <debug_mode> <parts_of_speech_flag>
```

---
**Please cite our work:**
```
@misc{
      evuru2024coda,
      title={CoDa: Constrained Generation based Data Augmentation for Low-Resource NLP}, 
      author={Chandra Kiran Reddy Evuru and Sreyan Ghosh and Sonal Kumar and Ramaneswaran S and Utkarsh Tyagi and Dinesh Manocha},
      year={2024},
      eprint={2404.00415},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```