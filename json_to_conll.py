import pandas as pd
import json
import re
import os
import argparse

parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
parser.add_argument('--dataset', '-d', type=str, help='Path to the input file')
parser.add_argument('--split', '-s', type=str, help='Path to the output file')
args = parser.parse_args()

out_conll_file = open(f"./tsv_data/out_data/{args.dataset}/{args.dataset}_{args.split}.conll", 'w')

df_const = pd.read_csv(f"./tsv_data/out_data/{args.dataset}/{args.dataset}_{args.split}_final_constraint.tsv",sep="\t",header=0)

# Specify the file path
file_path = f'./generation_data/{args.dataset}_{args.split}_abs_prompt_{{}}/generated_predictions.jsonl'
solo_file_path = f'./generation_data/{args.dataset}_{args.split}_solo_constraint/generated_predictions.jsonl'

new_rows = []

missing_count = 0

ner_phrase_tags_list = []
ner_phrases_list = []

for _, row in df_const.iterrows():
    ner_phrase_tags_list = ner_phrase_tags_list + list(eval(row["ner_phrase_tags"]))
    ner_phrases_list = ner_phrases_list + list(eval(row["ner_phrases"]))

def is_matching(word_list, phrase_list):
    word_list = [word.lower() for word in word_list]
    phrase_list = [word.lower() for word in phrase_list]
    return word_list == phrase_list

def get_phrase_idx_in_text(word_list, phrase):
    phrase_words = phrase.split()
    # Initialize start and end indexes
    start_index = None
    end_index = None

    ner_idxs = []
    for i in range(len(word_list)):
        if is_matching(word_list[i:i + len(phrase_words)], phrase_words):
            start_index = i
            end_index = i + len(phrase_words) - 1
            ner_idxs.append([start_index, end_index])

    if len(ner_idxs):
        return True, ner_idxs
    else:
        return False, None

def get_conll_str(text, idx, df, missing_count, ner_phrase_tags_list, ner_phrases_list):
    ner_tags = list(eval(df.at[idx, "ner_phrase_tags"]))
    ner_phrases = list(eval(df.at[idx, "ner_phrases"]))
    
    # Split the sentence by commas, periods, and spaces while keeping the delimiters
    word_list = re.split(r'([,.\s])', text)

    # Remove empty strings and spaces from the resulting list
    word_list = [word for word in word_list if word.strip()]

    tag_list = ["O"]*len(word_list)

    for phrase, tag in zip(ner_phrases, ner_tags):
        is_present, ner_idxs = get_phrase_idx_in_text(word_list, phrase)
        if is_present==True:
            act_tag = tag.split("-")[-1]
            for start_idx, end_idx in ner_idxs:
                tag_list[start_idx] = tag
                for k in range(start_idx+1,end_idx+1):
                    tag_list[k] = "I-" + act_tag
        else:
            missing_count = missing_count + 1
    conll_str = ""

    for word, tag in zip(word_list, tag_list):
        conll_str = conll_str + word + "\t" + tag + os.linesep

    return conll_str, missing_count

with open(solo_file_path, 'r') as file:
    i = 0
    count = 0
    for line in file:
        obj = json.loads(line)
        # print(i)
        predict_split = obj["predict"].split(":")
        if len(predict_split) > 1:
            predict_split = predict_split[1].split("\n\n")
            for text in predict_split:
                if len(text.strip())>0:
                    predict_split = text
                    break
        else:
            predict_split = obj["predict"] 
        conll_str, missing_count = get_conll_str(str(predict_split).strip('"'), int(obj["label"]), df_const, missing_count, ner_phrase_tags_list, ner_phrases_list)
        new_rows.append(conll_str)

print(f"Len of new data: {len(new_rows)}")

for row in new_rows:
    out_conll_file.write(row)
    out_conll_file.write(os.linesep)

out_conll_file.close()

print(f"Missing ner counts from generations : {missing_count}")