import pandas as pd
from tqdm.auto import tqdm
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from sentence_transformers import SentenceTransformer, util
import traceback
from datasets import load_dataset

import random

# import spacy
# spacy.prefer_gpu()

import sys
sys.path.append("../Lexical-Substitution/")

from lexsub_dropout import lexsub_dropout
# from lexsub_concatenation import lexsub_concatenation

import argparse
import os
import json
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import pandas as pd
import transformers
import torch.nn.functional as F

transformers.set_seed(42)
torch.backends.cudnn.deterministic = True

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#srun --pty --gres=gpu:rtxa6000:4 -c 8 --mem=128G -q high --time=1-00:00:00 bash

def remove_special_characters(text):
    pat = r'[^a-zA-z0-9.\-\_,!?/:;\"\'\s]'
    new_text =  re.sub(pat, '', text)
    new_text = new_text.replace('\n',' ').replace('\t',' ').replace('  ',' ').replace('\'s','').replace('\\n',' ').replace('\\t',' ')
    return new_text

def clean_pipeline(text):
    return remove_special_characters(text)

def get_that_sentence(text, that):
    text = '.' + text + '.'
    that = that.replace('(','\(')
    that = that.replace(')','\)')
    that = that.replace('([','\[')
    that = that.replace(']','\]')
    that = that.replace('$','\$')
    that = that.replace('.','\.')
    return re.findall(r'[\.?!]([^\.?!]*?%s[^\.?!]*?)[\.?!]'%that, text)

class ConstraintGeneration:
    def __init__(self, ds_name, inp_path, json_path, out_path, out_file, debug):
        if debug==1:
            self.df = pd.read_csv(inp_path,sep="\t",header=None).head(20)
        else:
            self.df = pd.read_csv(inp_path,sep="\t",header=None)
        self.word_regex = re.compile(r'^[A-Za-z]+$')
        self.out_pth = out_path
        self.out_file = out_file
        self.ds_name = ds_name
 
    def lexical_constraint(self, sentence, word_list, answer_sent):
        nv_list = word_list
        random.shuffle(nv_list)
        nv_final = []
        alt_map = {}
        alt_kws_list = []
        for word in nv_list:
        #     if self.word_regex.match(word) and word not in stopwords.words('english') and word not in answer and random.randint(0,1) == 1:
        #         alt_list = lexsub_dropout(sentence, word)
        #     else:
        #         alt_list = []

            # if len(alt_list) > 1 and len(alt_list[0][0]) > 0 and len(alt_list[0][0].strip()) > 1 and word not in answer and  self.word_regex.match(alt_list[0][0]):
            #     alt_map[word] = alt_list[0][0]
            if word not in stopwords.words('english'):
                nv_final.append(word)
        nv_final = list(set(nv_final))
        nv_final_lem = nv_final

        # for idx, _ in enumerate(nv_final):
        #     if nv_final[idx] in alt_map:
        #         alt_kw = alt_map[nv_final[idx]]
        #         if random.randint(0,1) == 1 and (alt_kw not in nv_final_lem) and word not in answer:
        #             nv_final[idx] = nv_final[idx] + " or " + alt_kw
        #         if (alt_kw not in nv_final_lem):
        #             alt_kws_list.append(alt_kw)

        # print(nv_final)
        # print(nv_final_lem)
        # print(alt_kws_list)

        # rand_other_kws = random.sample(other_list, k=random.randint(1, int(max(1, 0.3*len(other_list))))) if len(other_list)>0 else []

        rand_nv_kws = random.sample(nv_final, k=random.randint(1, int(max(1,0.3*len(nv_final))))) if len(nv_final)>0 else []
        # print(get_that_sentence(sentence, answer)[0])
        print(answer_sent)
        inc_kws  = (((",".join(rand_nv_kws)) + ",") if len(rand_nv_kws) > 0 else "") + f'"{answer_sent}"'
        exc_kws = ""

        # if len(alt_kws_list) > 0:
        #     rand_neg = random.sample(alt_kws_list, k=random.randint(1, int(max(1,0.3*len(alt_kws_list)))))
        #     exc_kws = ",".join(rand_neg)
            # constraint = f"Write something with the following keywords: {inc_kws}, but do not include the following keywords: {exc_kws}"
            # print(constraint)
            # print()
        # else:
        #     constraint = f"Write something with the following keywords: {inc_kws}"
        inc_kws = inc_kws.rstrip(",")
        exc_kws = exc_kws.rstrip(",")
        return inc_kws, exc_kws

    def generate_constraints(self):

        raw_datasets = load_dataset("squad")

        gold_ids = list(set(self.df.iloc[:, 1].tolist()))

        raw_datasets["train"] = raw_datasets["train"].filter(lambda example: example["id"] in gold_ids)

        print(raw_datasets)
        self.df = pd.DataFrame.from_dict(raw_datasets["train"])

        prompt_inc_exc =[
"""Write a brief document with multiple sentences corresponding to the following constraints:
1. The document should have the following keywords: {inc_kws}, but should not have the following keywords : {exc_kws}.
2. The document should anwer the question: {question}."""
        ]

        prompt_inc = [
"""Write a brief document with multiple sentences corresponding to the following constraints:
1. The document should have the following keywords: {inc_kws}.
2. The document should anwer the question: {question}."""
        ]

        inc_kws_list = []
        exc_kws_list = []
        pos_seq_list = []
        topic_list = []
        min_words_list = []
        max_words_list = []
        num_sents_list = []
        sent_key_list = []
        abs_const_list = []

        final_const_list = []
        final_const_topic_list = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Generating constraints", unit="Rows/s"):
            # sent = clean_pipeline(row["context"])
            sent = row["context"]
            print(f"Sent: {sent}")
            abs_prompt = "Write an abstract of the following document in a few words:\n{}".format(sent)
            abs_const_list.append(abs_prompt)

            # pos_sequence = self.get_spacy_data(sent)
            # pos_seq_list.append(pos_sequence)

            # top_k_kws = self.get_top_k_kws(sent, row["label"])

            # rand_exemplars= self.get_exemplars(row["text"], sent, row["label"])

            # print(f"Random exemplars: {rand_exemplars}")
            # print(f"Idx: {idx}, len of top_k_kws: {top_k_kws}")
            # print(f"top_k_kws: {top_k_kws}")
            # if idx == 1:
            #     exit()
            # continue
            # pos_seq_list.append(pos_sequence)

            num_words = len(sent.split())
            min_len = max(int(num_words),5)
            max_len = max(int((3*num_words)/2),10)
            min_words_list.append(min_len)
            max_words_list.append(max_len)

            # topic = row["label"]
            # topic_list.append(topic)

            num_sents = len(sent_tokenize(sent))
            num_sents_list.append(num_sents)

            sent_key = "sentence" if num_sents == 1 else "sentences"
            sent_key_list.append(sent_key)

            try:
                # print("Doing conc_lexical_constraint.")
                answer_sent = get_that_sentence(sent, row["answers"]["text"][0])[0]
                print(answer_sent)
                inc_kws, exc_kws = self.lexical_constraint(sent, sent.replace(answer_sent,"").split(" "), answer_sent)
                inc_kws = inc_kws
                print(f"Inc : {inc_kws}")
                print(f"Exc : {exc_kws}")
                inc_kws_list.append(inc_kws)
                exc_kws_list.append(exc_kws)
            except Exception as ex:
                print(ex)
                print(traceback.print_exc())
                inc_kws, exc_kws = "", ""
                inc_kws_list.append(inc_kws)
                exc_kws_list.append(exc_kws)

            curr_prompt = []

            inc_exc = False
            inc = False

            if len(inc_kws) > 0 and len(exc_kws) > 0:
                curr_prompt = prompt_inc_exc
                inc_exc = True
            else:
                curr_prompt = prompt_inc

            for _, prompt in enumerate(curr_prompt):
                if inc_exc:
                    prompt = prompt.format(
                        # min_words=min_len,
                        # max_words=max_len,
                        question=row["question"],
                        inc_kws=inc_kws,
                        exc_kws=exc_kws
                    )
                else:
                    prompt = prompt.format(
                        # min_words=min_len,
                        # max_words=max_len,
                        question=row["question"],
                        inc_kws=inc_kws
                    )
                final_const_topic_list.append(prompt) 

        self.df["inc_kws"] = inc_kws_list
        self.df["exc_kws"] = exc_kws_list
        # self.df["pos_seq"] = pos_seq_list
        self.df["min_words"] = min_words_list
        self.df["max_words"] = max_words_list
        self.df["num_sents"] = num_sents_list
        self.df["abs_const"] = abs_const_list

        self.df["final_topic_const"] = final_const_topic_list

        self.df.to_csv(os.path.join(self.out_pth,self.out_file+".tsv"),sep="\t",index=False)

        self.save_json_abstract()

    def save_json_abstract(self):
        json_data = []

        for _, row in self.df.iterrows():
            json_entry = {
                "instruction": row["abs_const"],
                "output": row["answers"]["text"][0],
                "input": ""
            }
            json_data.append(json_entry)

        # Write the JSON data to a file
        with open(os.path.join(self.out_pth,self.out_file+"_abs.json"), 'w') as json_file:
            json.dump(json_data, json_file, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-j', '--json_path', type=str, required=True)
    parser.add_argument('-p', "--par_dir", type=str, required=True)
    parser.add_argument('-ds', "--ds_dir", type=str, required=True)
    parser.add_argument('-o', '--out_file', type=str, required=True)  
    parser.add_argument('-d', '--debug', type=int, default=0)    
    args = parser.parse_args()
    print(f"Main debug value : {args.debug}")
    os.makedirs(args.par_dir, exist_ok=True)
    out_dir = os.path.join(args.par_dir, args.ds_dir)
    os.makedirs(out_dir, exist_ok=True)
    cg = ConstraintGeneration(args.ds_dir, args.input_path, args.json_path, out_dir, args.out_file, args.debug)  
    cg.generate_constraints()

if __name__ == "__main__":
    main()