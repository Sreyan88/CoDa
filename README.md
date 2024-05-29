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
@inproceedings{
      evuru2024coda,
      title={CoDa: Constrained Generation based Data Augmentation for Low-Resource {NLP}},
      author={Chandra Kiran Reddy Evuru and Sreyan Ghosh and Sonal Kumar and Ramaneswaran S and Utkarsh Tyagi and Dinesh Manocha},
      booktitle={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
      year={2024},
      url={https://openreview.net/forum?id=O5jNMEmc41}
}
```
