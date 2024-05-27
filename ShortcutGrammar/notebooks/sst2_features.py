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

from src.utils import tree_utils
from src.features import feature_utils

data = feature_utils.load_trees_from_predictions(
    output_dir="./ShortcutGrammar/output/sst2", 
    dataset="sst2"
)

# print(data)

# print(data["train"]["tree"][0][0])

feature_utils.add_features(data, "Subtrees", feature_utils.pcfg_subtrees_by_depth(7))
subtrees = feature_utils.get_subtree_feature_table(data)
subtrees.head(10)


feature_utils.add_merges(data, "Subtrees", K=2000, merge_name="Subtree groups",
                         filter_=lambda w: w[0].count("*") >= 2)

df = feature_utils.get_merged_feature_table(data, merge_name="Subtree groups").head(30)

# print(df)

df.to_csv("test.csv",sep=",",index=False)
# print("Saved csv")

k = 3

negative = df[df['Majority label'] == 'Negative']
positive = df[df['Majority label'] == 'Positive']

top_k_negative = negative.sort_values('MI', ascending=False).head(k)['Root'].to_list()
top_k_positive = positive.sort_values('MI', ascending=False).head(k)['Root'].to_list()

print('Negative:')
print(top_k_negative)

print('\nPositive:')  
print(top_k_positive)

# for root in top_k_negative:
#     feature_utils.add_merges(data, "Subtrees", merge_name=f"Root {root}", K=1000, by_template=True,
#                          filter_=lambda w: w[0].startswith(f"({root} "))
    
# for root in top_k_negative:
#     feature_utils.add_merges(data, "Subtrees", merge_name=f"Root {root}", K=1000, by_template=True,
#                          filter_=lambda w: w[0].startswith(f"({root} "))

exit()

feature_utils.add_merges(data, "Subtrees", merge_name="Root 29", K=1000, by_template=True,
                         filter_=lambda w: w[0].startswith("(29 "))

df = feature_utils.get_merged_feature_table(data, merge_name="Root 29").head(20)

df.to_csv("test_29.csv",sep=",",index=False)

k=2

for label in df['Majority label'].unique():

    # Filter by label 
    label_df = df[df['Majority label'] == label]

    # Sort by '% majority' and get top k
    topk = label_df.sort_values('MI', ascending=False).head(k)

    print(f"\n{label}:")
    for i, row in topk.iterrows():
        print(f"{row['Root']}")
        print(f"{row['MI']}")
        print(f"{row['Examples']}")
        print()

    print("****")
    print()
    print()