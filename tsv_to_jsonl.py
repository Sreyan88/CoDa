import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
parser.add_argument('--dataset', '-d', type=str, help='Path to the input file')
parser.add_argument('--split', '-s', type=str, help='Path to the output file')
args = parser.parse_args()

df = pd.read_csv(f"/fs/nexus-projects/audio-visual_dereverberation/ck_naacl/tsv_data/out_data/{args.dataset}/{args.dataset}_{args.split}_final_constraint.tsv",sep="\t",header=0)

# Create a list of dictionaries in the desired format
json_data = []
for idx, row in df.iterrows():
    json_entry = {
        "instruction": row["final_topic_const"],
        "input": row["label"],
        "output": row["label"]
    }
    json_data.append(json_entry)

# Write the JSON data to a file
with open(f'/fs/nexus-projects/audio-visual_dereverberation/ck_naacl/tsv_data/out_data/{args.dataset}/{args.dataset}_{args.split}_solo_constraint.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=2)