import os
if os.getcwd().endswith("notebooks"):
    os.chdir("..")
import sys
sys.path.append("./ShortcutGrammar")
import importlib
import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", None)
pd.set_option('display.max_rows', 500)
import json
from src.utils import tree_utils
from src.features import feature_utils
import argparse

def get_kws_list(df, labels, k):
    top_k_kws_dict = {}
    for label in labels:
        top_k_kws_list = []
        label_group = df[df['Majority label'] == label.lower()]
        top_k = label_group.sort_values('MI', ascending=False).head(2*k)
        for _, row in top_k.iterrows():
            top_k_kws_list.append(list(filter(lambda x: "[UNK]" not in x and len(x.lstrip().rstrip()) > 0, [y.lstrip().rstrip() for y in (row["Examples"].split(","))])))
        top_k_kws_dict[label] = top_k_kws_list
    return top_k_kws_dict

def get_exemplars(df, label, kws_list):
    exemplar_list = []
    filt_df = df[df["label_name"] == label]
    for kws in kws_list:
        exemplars = {}
        for kw in kws:
            for _, row in filt_df.iterrows():
                if kw.replace(" \' ","\'") in row["sentence"]:
                    if kw in exemplars:
                        break
                    else:                
                        exemplars[kw] = row["sentence"]
        exemplar_list.append(exemplars)
    exemplar_list = [exemplars for exemplars in exemplar_list if len(exemplars) > 1]
    return exemplar_list

def main(args):

    name = args.dataset

    data = feature_utils.load_trees_from_predictions(
        output_dir=f"./ShortcutGrammar/output/{name}/", 
        dataset=name
    )

    feature_utils.add_features(data, "Subtrees", feature_utils.pcfg_subtrees_by_depth(20))
    subtrees = feature_utils.get_subtree_feature_table(data)

    feature_utils.add_merges(data, "Subtrees", K=100000, merge_name="Subtree groups")
    subtree_groups = feature_utils.get_merged_feature_table(data, merge_name="Subtree groups")
    df = subtree_groups.sort_values(by=["MI"], ascending=False)
    df['Majority label'] = df["Majority label"].apply(lambda x: x.lower())
    k = args.top_k

    ori_df = pd.read_csv(f"./ShortcutGrammar/data/{args.dataset}/test.tsv", sep="\t", header=0)

    labels_list = list(ori_df['label_name'].unique())
    print(f"Labels list : {labels_list}")

    top_k_kws_dict = get_kws_list(df, labels_list, k)
    print(top_k_kws_dict)
    # print()
    # print()
    # get_exemplars(ori_df, "Education & Reference", top_k_kws_dict["Education & Reference"])
    # exit()
    top_k_exemplars_dict = {}
    for key, value in top_k_kws_dict.items():
        top_k_exemplars = get_exemplars(ori_df, key, value)
        top_k_exemplars_dict[key] = top_k_exemplars
    # print(top_k_exemplars_dict.keys())

    json_data = []

    for label in labels_list:
        label_prompt = []
        print(label)
        top_k_exemplars = top_k_exemplars_dict[label]
        print(top_k_exemplars)
        print()
        print()
        for i in range(0, k):
            prompt = """I will provide you with some constraints followed by pairs of sentences and their respective keywords. Please do the following:
1. Find one high level concept common across these sentences considering their respective keywords.
2. The high level concept should contain at most 2 words.
3. The high level concept should be enclosed in #.
4. Output only the high level concept in the above format and nothing else.
"""
            print(top_k_exemplars)
            print(len(top_k_exemplars))
            print(i)
            label_sent_list = list(top_k_exemplars[i].values())[:k]
            label_kws_list = list(top_k_exemplars[i].keys())[:k]
            print(f"top-k : {i}")
            print(len(label_sent_list))
            print(len(label_kws_list))
            print()
            print()

            for j in range(0, min(len(label_kws_list), k)):
                prompt = prompt + "{" + f"sentence: {label_sent_list[j]}, keyword: {label_kws_list[j]}" + "}" + os.linesep

            label_prompt.append(prompt)

        for idx in range(0, k):
            json_entry = {
                "instruction": label_prompt[idx],
                "input": "",
                "output": label
            }
            json_data.append(json_entry)

    file_name = f"{name}_{args.split}_concept_constraint"

    with open(f"./LLaMA-Factory/data/{file_name}.json", "w") as f:
        json.dump(json_data, f, indent=2)

    with open("./LLaMA-Factory/data/dataset_info.json", "r") as json_file:
        data = json.load(json_file)

    data[f"{file_name}"] = {"file_name": f"{file_name}.json"}

    with open("./LLaMA-Factory/data/dataset_info.json", "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
    parser.add_argument('--dataset', '-d', required=True, type=str, help='Path to the input file')
    parser.add_argument('--top_k', '-k', default=3, type=int, help='Path to the input file')
    parser.add_argument('--split', '-s', default="100", type=str, help='Path to the input file')
    args = parser.parse_args()
    main(args)