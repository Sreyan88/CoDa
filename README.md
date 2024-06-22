# CoDa : Constrained Generation based Data Augmentation for Low-Resource NLP

Implementation of [CoDa : Constrained Generation based Data Augmentation for Low-Resource NLP](https://arxiv.org/pdf/2404.00415)

![Proposed Methodology](./diagram.png)

### Constraint Generation:

1. For Classification tasks:

```shell
sh classification_pipeline.sh <dataset_name> <dataset_split> <debug_mode> <dataset_split_for_shortcut_grammar>
```

<debug_mode> - generate augmentations for only the first 10 entries in the dataset.  
<dataset_split> - The low-resource split of the datasets (e.g., 100, 200, 500 and 1000).  
<dataset_split_for_shortcut_grammar> - The split of the dataset used for finding shortcuts as described in the paper. Use either train/dev/test.  
<dataset_name> - Name of the dataset to be used for generation.  

Datasets currently supported:  
Huffpost  
Yahoo  
OTS  
ATIS  
Massive  

Example command:

```shell
sh classification_pipeline.sh huff 500 0 test
```

2. For NER tasks:

```shell
sh ner_pipeline.sh <dataset_name> <dataset_split> <debug_mode> <parts_of_speech_flag>
```
<parts_of_speech_flag> - whether to generate augmentations with parts of speech constraint.

Example:

```shell
sh ner_pipeline.sh conll2003 500 0 0
```

Datasets currently supported:  
CoNLL-2003  
OntoNotes  
EBMNLP  
BC2GM  


### Training & Evaluation:
The scripts in the previous section generate synthetic augmentations, add original data, and place the combined data in `tsv_data/out_data`. The model can be trained further on the original + synthetic data file and evaluated on the test split of the input dataset.  

Note: If you find a hard time setting up the environment, just raise an issue!


### Acknowledgments:  

We use the following repositories to implement our methodology:
1. [Lexical-Substitution](https://github.com/jvladika/Lexical-Substitution)
2. [ShortcutGrammar](https://github.com/princeton-nlp/ShortcutGrammar)

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
