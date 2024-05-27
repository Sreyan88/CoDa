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

def get_kws_list(df, label, k):
    label_group = df[df['Majority label'] == label]
    
    label_group.sort_values('MI', ascending=False).to_csv(f"./ShortcutGrammar/notebooks/{label}.csv",index=False)
    
    top_k = label_group.sort_values('MI', ascending=False).head(k)

    print(top_k["Examples"].to_list()[0])

    top_k_kws = []

    for _, row in top_k.iterrows():
        top_k_kws.append(list(filter(lambda x:  "[UNK]" not in x, [y.lstrip().rstrip() for y in (row["Examples"].split(","))])))

    return top_k_kws

def get_exemplars(df, kws_list):
    exemplar_list = []
    for kws in kws_list:
        exemplars = {}
        for kw in kws:
            for _, row in df.iterrows():
                if kw in row["text"]:
                    if kw in exemplars:
                        continue
                    else:                
                        exemplars[kw] = row["text"]
        exemplar_list.append(exemplars)
    return exemplar_list

name = "sst2"

data = feature_utils.load_trees_from_predictions(
    output_dir="./ShortcutGrammar/output/" + name, 
    dataset=name
)

# print(data)

# print(data["train"]["tree"][0][0])

feature_utils.add_features(data, "Subtrees", feature_utils.pcfg_subtrees_by_depth(20))
subtrees = feature_utils.get_subtree_feature_table(data)

feature_utils.add_merges(data, "Subtrees", K=100000, merge_name="Subtree groups")
subtree_groups = feature_utils.get_merged_feature_table(data, merge_name="Subtree groups")
df = subtree_groups.sort_values(by=["MI"], ascending=False)

# df.to_csv("test.csv",sep=",",index=False)
# print("Saved csv")

k = 3

top_k_pos_kws_list = get_kws_list(df, "Positive", k)
top_k_neg_kws_list = get_kws_list(df, "Negative", k)

# print(top_k_pos_kws_list)

ori_df = pd.read_csv("./ShortcutGrammar/data/low_res/sst2_100.tsv", sep="\t", header=0)

top_k_pos_exemplars = get_exemplars(ori_df, top_k_pos_kws_list)

top_k_neg_exemplars = get_exemplars(ori_df, top_k_neg_kws_list)

# print(top_k_pos_exemplars)

# prompt = """Given sentences abstractify the role the keywords play in the respective sentence into a concept of atmost one word:
# Sentences: {sentences}
# Keywords: {keywords}
# """

# prompt = """I will give sentences and their respective keywords. Find an abstract theme/concept across these sentences considering their keywords and output only the answer in a single word, with the following format Concept:answer. -
# Sentences: {sentences}
# Keywords: {keywords}"""


prompt = """I will give pairs of sentences and their respective keywords. Find an abstract theme/concept across these sentences considering their keywords and output only the answer in a single word, with the following format Concept:answer. -
"""

# I will give sentences and their respective keywords. Find an abstract theme/concept across these sentences considering their keywords and output only the answer in a single word, with the following format Concept:answer. -
# Sentences: ["a testament to the film 's considerable charm ", "a thriller with an edge -- which is to say that it does n't follow the stale , standard , connect-the-dots storyline which has become commonplace in movies that explore the seamy underbelly of the criminal world . ", "it 's refreshing that someone understands the need for the bad boy ; diesel , with his brawny frame and cool , composed delivery , fits the bill perfectly "]
# Keywords: ['film', 'standard', 'boy']

pos_prompt = []
neg_prompt = []
json_data = []

for i in range(0,k):
    pairs = ""
    prompt_pos = """I will provide you with some constraints followed by pairs of sentences and their respective keywords. Please do the following:
1. Find one high level concept common across these sentences considering their respective keywords.
2. The high level concept should contain at most 2 words.
3. The high level concept should be enclosed in #.
4. Output only the high level concept in the above format and nothing else.
    """
    pos_sent_list = list(top_k_pos_exemplars[i].values())[:k]
    pos_kws_list = list(top_k_pos_exemplars[i].keys())[:k]

    prompt_neg = """I will provide you with some constraints followed by pairs of sentences and their respective keywords. Please do the following:
1. Find one high level concept common across these sentences considering their respective keywords.
2. The high level concept should contain at most 2 words.
3. The high level concept should be enclosed in #.
4. Output only the high level concept in the above format and nothing else.
    """
    neg_sent_list = list(top_k_neg_exemplars[i].values())[:k]
    neg_kws_list = list(top_k_neg_exemplars[i].keys())[:k]
    
    for j in range(0,k):
        prompt_pos = prompt_pos + "{" + f"sentence: {pos_sent_list[j]}, keyword: {pos_kws_list[j]}" + "}" + os.linesep
        prompt_neg = prompt_neg + "{" + f"sentence:{neg_sent_list[j]}, keyword:{neg_kws_list[j]}" + "}" + os.linesep
    


    pos_prompt.append(prompt_pos)
    neg_prompt.append(prompt_neg)

for idx in range(0,k):
    # pos_prompt.append(prompt.format(sentences=list(top_k_pos_exemplars[idx].values())[:k], keywords=list(top_k_pos_exemplars[idx].keys())[:k]))
    json_entry = {
        "instruction": pos_prompt[idx],
        "input": "",
        "output": "positive"
    }
    json_data.append(json_entry)

for idx in range(0,k):
    # pos_prompt.append(prompt.format(sentences=list(top_k_pos_exemplars[idx].values())[:k], keywords=list(top_k_pos_exemplars[idx].keys())[:k]))
    json_entry = {
        "instruction": neg_prompt[idx],
        "input": "",
        "output": "negative"
    }
    json_data.append(json_entry)

file_name = "sst2_concept_constraint"

with open(f"./LLaMA-Factory/data/{file_name}.json", "w") as f:
    json.dump(json_data, f, indent=2)

with open("./LLaMA-Factory/data/dataset_info.json", "r") as json_file:
    data = json.load(json_file)

data[f"{file_name}"] = {"file_name": f"{file_name}.json"}

with open("./LLaMA-Factory/data/dataset_info.json", "w") as f:
    json.dump(data, f, indent=2)

# prompt = prompt.format(sentences=list(top_k_pos_exemplars[1].values())[:k], keywords=list(top_k_pos_exemplars[1].keys())[:k])

# print(pos_prompt)

# print(neg_prompt[-1])

# exit()