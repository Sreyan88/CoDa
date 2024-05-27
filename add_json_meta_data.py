import json
import argparse

parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
parser.add_argument('--dataset', '-d', type=str, help='Path to the input file')
parser.add_argument('--split', '-s', type=str, help='Path to the output file')
args = parser.parse_args()

with open("./generation_data/dataset_info.json", "r") as json_file:
    data = json.load(json_file)

for suffix in ["_concept_constraint", "_final_constraint_abs", "_solo_constraint", "_abs_prompt_1", "_abs_prompt_2"]:
    file_name = f"{args.dataset}_{args.split}{suffix}"
    data[f"{file_name}"] = {"file_name": f"{file_name}.json"}

with open("./generation_data/dataset_info.json", "w") as f:
    json.dump(data, f, indent=2)