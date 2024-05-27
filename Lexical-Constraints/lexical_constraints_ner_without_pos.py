import pandas as pd
from tqdm.auto import tqdm
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

import random
random.seed(42)

import spacy
spacy.prefer_gpu()

import sys
sys.path.append("../Lexical-Substitution/")

from lexsub_dropout import lexsub_dropout
# from lexsub_concatenation import lexsub_concatenation

import argparse
import os
import json
from transformers import AutoTokenizer, AutoModel
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

def remove_special_characters(text):
    pat = r'[^a-zA-z0-9.\-\_,!?/:;\"\'\s]'
    new_text =  re.sub(pat, '', text)
    new_text = new_text.replace('\n',' ').replace('\t',' ').replace('  ',' ').replace('\'s','').replace('\\n',' ').replace('\\t',' ')
    return new_text

def clean_pipeline(text):
    return remove_special_characters(text)

class ConstraintGeneration:
    def __init__(self, ds_name, inp_path, out_path, out_file, debug):
        self.nlp = spacy.load("en_core_web_lg")
        if debug==1:
            self.df = pd.read_csv(inp_path,sep="\t",header=0).head(10)
        else:
            self.df = pd.read_csv(inp_path,sep="\t",header=0)
        self.word_regex = re.compile(r'^[A-Za-z]+$')
        self.out_pth = out_path
        self.ds_name = ds_name
        self.out_file = out_file
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # self.model_name = "sentence-transformers/sentence-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
    def lexical_constraint(self, sentence, word_list, ner_list):
        word_list_len = len(word_list)
        i = 0
        inc_kws = []
        ner_phrase = ""
        nv_list = []
        print(ner_list)
        begin_label = ""
        ner_tag_list = []
        ner_phrase_list = []
        while i < word_list_len:
            word = word_list[i]
            if ner_list[i] == "O":
                if len(ner_phrase) >0:
                    ner_phrase_list.append(ner_phrase.rstrip(" ").lstrip(" "))
                    inc_kws.append(ner_phrase.rstrip(" ").lstrip(" "))
                    ner_tag_list.append(begin_label)
                    ner_phrase = ""
                    begin_label = ""
                if self.word_regex.match(word) and word not in stopwords.words('english'):
                    nv_list.append(word)
            else:
                if ner_list[i].startswith("B-"):
                    begin_label = ner_list[i]
                ner_phrase = ner_phrase + word_list[i] + " "
            i = i + 1
            if i == word_list_len and len(ner_phrase) > 0:
                inc_kws.append(ner_phrase.rstrip(" ").lstrip(" "))
                ner_phrase = ""

        nv_final = []
        alt_map = {}
        alt_kws_list = []
        for word in nv_list:
            if word in word_list and self.word_regex.match(word) and random.randint(0,1) == 1:
                alt_list = lexsub_dropout(sentence, word)
            else:
                alt_list = []

            if len(alt_list) > 1 and len(alt_list[0][0]) > 0 and len(alt_list[0][0].strip()) > 1 and self.word_regex.match(alt_list[0][0]):
                alt_map[word] = alt_list[0][0]

            nv_final.append(word)

        for idx, _ in enumerate(nv_final):
            if nv_final[idx] in alt_map:
                alt_kw = alt_map[nv_final[idx]]
                if random.randint(0,1) == 1:
                    nv_final[idx] = nv_final[idx] + " or " + alt_kw
                elif (alt_kw not in nv_final):
                    alt_kws_list.append(alt_kw)

        print(nv_final)
        print(alt_kws_list)
        
        rand_nv_kws = random.sample(nv_final, k=random.randint(1, int(max(1,0.3*len(nv_final))))) if len(nv_final)>0 else []


        inc_kws  = (",".join(inc_kws) + "," + ",".join(rand_nv_kws)).rstrip(",").lstrip(",")
        exc_kws = ""

        if len(alt_kws_list) > 0:
            rand_neg = random.sample(alt_kws_list, k=random.randint(1, int(max(1,0.3*len(alt_kws_list)))))
            exc_kws = (",".join(rand_neg)).rstrip(",").lstrip(",")

        print(f"inc_kws: {inc_kws}")
        print(f"exc_kws: {exc_kws}")
        print(f"ner_tag_list: {ner_tag_list}")

        return inc_kws, exc_kws, ner_phrase_list, ner_tag_list

    def get_spacy_data(self, sent):
            doc = self.nlp(sent)
            pos_list = []

            for token in doc:
                pos_list.append(token.pos_)

            pos_sequence = " ".join(pos_list)

            return pos_sequence

    def get_exemplars(self, sent_ori):
        label_sent_list = [x for x in self.df["text"].tolist() if x!=sent_ori]
        rand_sent_list = random.sample(label_sent_list, min(len(label_sent_list), 3))
        random.shuffle(rand_sent_list)
        return ','.join(['\"{}\"'.format(sentence) for sentence in rand_sent_list])

    def generate_constraints(self):

        ner_tag_to_keyword_dict = {
            "conll2003": {
                'O': 'other', 
                'B-MISC': 'miscellaneous',
                'I-MISC': 'miscellaneous',
                'B-PER': 'person',
                'I-PER': 'person', 
                'B-LOC': 'location', 
                'I-LOC': 'location',
                'B-ORG': 'organization', 
                'I-ORG': 'organization'
            },
            "multiconer": {
                'O': 'other', 
                'B-PER': 'person',
                'I-PER': 'person', 
                'B-LOC': 'location', 
                'I-LOC': 'location',
                'B-GRP': 'group', 
                'I-GRP': 'group', 
                'B-CW': 'creative work', 
                'I-CW': 'creative work',
                'B-CORP': 'corporation',
                'I-CORP': 'corporation',
                'B-PROD': 'product',
                'I-PROD': 'product'
            },
            "onto": {
                'O': 'other', 
                'B-DATE': 'date', 
                'I-DATE': 'date', 
                'B-MONEY': 'money', 
                'I-MONEY': 'money', 
                'B-WORK_OF_ART': 'work of art',
                'I-WORK_OF_ART': 'work of art',
                'B-CARDINAL': 'cardinal', 
                'I-CARDINAL': 'cardinal',
                'B-ORG': 'organization', 
                'I-ORG': 'organization', 
                'B-PERSON': 'person', 
                'I-PERSON': 'person', 
                'B-GPE': 'geo-political entity',
                'I-GPE': 'geo-political entity', 
                'B-NORP': 'affiliation', 
                'B-PERCENT': 'percent', 
                'I-PERCENT': 'percent', 
                'B-ORDINAL': 'ordinal',  
                'B-TIME': 'time', 
                'I-TIME': 'time', 
                'B-LOC': 'location', 
                'I-LOC': 'location', 
                'B-PRODUCT': 'product', 
                'B-FAC': 'building', 
                'I-FAC': 'building', 
                'B-EVENT': 'event', 
                'I-EVENT': 'event', 
                'B-QUANTITY': 'quantity', 
                'I-QUANTITY': 'quantity', 
                'B-LANGUAGE': 'language', 
                'I-NORP': 'affiliation', 
                'B-LAW': 'law', 
                'I-LAW': 'law', 
                'I-PRODUCT': 'product', 
                'I-LANGUAGE': 'language', 
                'I-ORDINAL': 'ordinal'           
            }
        }

        ner_tag_to_keyword_dict = ner_tag_to_keyword_dict[self.ds_name]

        prompt_inc_exc =[ 
"""Write a brief document with a single sentence or multiple sentences corresponding to the following abstract description: $abs$.
Additionally, the document should have the following contraints:
1. The document should have the following keywords: {inc_kws}, but should not have the following keywords : {exc_kws}.
2. {ner_labels}.
3. Here are also some examples: {exemplars}.
4. The document should have a length of {min_words}-{max_words} words."""
, 
"""Write a brief document with a single sentence or multiple sentences with the following constraints:
1. The document should have the following keywords: {inc_kws}, but should not have the following keywords : {exc_kws}.
2. {ner_labels}.
3. Here are also some examples: {exemplars}.
4. The document should have a length of {min_words}-{max_words} words."""
        ]

        prompt_inc = [
"""Write a brief document with a single sentence or multiple sentences corresponding to the following abstract description: $abs$.
Additionally, the document should have the following contraints:
1. The document should have the following keywords: {inc_kws}.
2. {ner_labels}.
3. Here are also some examples: {exemplars}.
4. The document should have a length of {min_words}-{max_words} words."""
,
"""Write a brief document with a single sentence or multiple sentences with the following constraints:
1. The document should have the following keywords: {inc_kws}.
2. {ner_labels}.
3. Here are also some examples: {exemplars}.
4. The document should have a length of {min_words}-{max_words} words."""
        ]

        prompt_none = [
"""Write a brief document with a single sentence or multiple sentences corresponding to the following abstract description: $abs$.
Additionally, the document should have the following contraints:
1. {ner_labels}.
2. Here are also some examples: {exemplars}.
3. The document should have a length of {min_words}-{max_words} words."""
,
"""Write a brief document with a single sentence or multiple sentences with the following constraints:
1. {ner_labels}. 
2. Here are also some examples: {exemplars}.
3. The document should have a length of {min_words}-{max_words} words."""
        ]

        inc_kws_list = []
        exc_kws_list = []
        pos_seq_list = []
        min_words_list = []
        max_words_list = []
        num_sents_list = []
        sent_key_list = []
        abs_const_list = []
        ner_tags_list = []
        ner_phrase_list = []

        final_const_list = []
        final_const_topic_list = []

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Generating constraints", unit="Rows/s"):
            sent = row["text"]
            
            abs_prompt = "Write an abstract of the following document in a few words:\n{}".format(sent)
            abs_const_list.append(abs_prompt)

            rand_exemplars= self.get_exemplars(row["text"])
            print(f"Random exemplars: {rand_exemplars}")

            pos_sequence = self.get_spacy_data(sent)

            pos_seq_list.append(pos_sequence)

            num_words = len(sent.split())
            min_len = max(int(num_words/2),5)
            max_len = max(int((3*num_words)/2),10)
            min_words_list.append(min_len)
            max_words_list.append(max_len)

            num_sents = len(sent_tokenize(sent))
            num_sents_list.append(num_sents)

            sent_key = "sentence" if num_sents == 1 else "sentences"
            sent_key_list.append(sent_key)

            word_list = list(eval(str(row["tokens"])))

            ner_list = list(eval(str(row["ner_tags"])))

            inc_kws, exc_kws, ner_phrases, ner_tags = self.lexical_constraint(sent, word_list, ner_list)
            inc_kws_list.append(inc_kws)
            exc_kws_list.append(exc_kws)
            ner_tags_list.append(ner_tags)
            ner_phrase_list.append(ner_phrases)

            ner_labels = ", ".join(f"{ner_phrase} is {ner_tag_to_keyword_dict[ner_tag]}" for ner_phrase, ner_tag in zip(ner_phrases, ner_tags))

            curr_prompt = []

            inc_exc = False
            inc = False

            if len(inc_kws) > 0 and len(exc_kws) > 0:
                curr_prompt = prompt_inc_exc
                inc_exc = True
            elif len(inc_kws) > 0:
                curr_prompt = prompt_inc
                inc = True
            else:
                curr_prompt = prompt_none

            for idx, prompt in enumerate(curr_prompt):
                if idx == 1:
                    if inc_exc:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            min_words=min_len,
                            max_words=max_len,
                            ner_labels=ner_labels,
                            inc_kws=inc_kws,
                            exc_kws=exc_kws
                        )
                    elif inc:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            min_words=min_len,
                            max_words=max_len,
                             ner_labels=ner_labels,
                            inc_kws=inc_kws
                        )
                    else:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            min_words=min_len,
                            max_words=max_len,
                            ner_labels=ner_labels
                        )       
                    final_const_topic_list.append(prompt) 
                else:
                    if inc_exc:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            min_words=min_len,
                            max_words=max_len,
                            ner_labels=ner_labels,
                            inc_kws=inc_kws,
                            exc_kws=exc_kws
                        )
                    elif inc:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            ner_labels=ner_labels,
                            min_words=min_len,
                            max_words=max_len,
                            inc_kws=inc_kws
                        )
                    else:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            ner_labels=ner_labels,
                            min_words=min_len,
                            max_words=max_len
                        ) 
                    final_const_list.append(prompt)

        self.df["inc_kws"] = inc_kws_list
        self.df["exc_kws"] = exc_kws_list
        self.df["pos_seq"] = pos_seq_list
        self.df["min_words"] = min_words_list
        self.df["max_words"] = max_words_list
        self.df["num_sents"] = num_sents_list
        self.df["abs_const"] = abs_const_list
        self.df["ner_phrase_tags"] = ner_tags_list
        self.df["ner_phrases"] = ner_phrase_list
        self.df["final_const"] = final_const_list
        self.df["final_topic_const"] = final_const_topic_list

        sim_df = self.df[["text"]]

        # List of input sentences
        sentences = sim_df["text"].to_list()


        # Tokenize the sentences and convert them to tensors in batches
        batched_tokens = self.tokenizer(sentences, padding=True, max_length=512, truncation=True, return_tensors="pt")

        # Pass the tokenized batch through the BERT model
        with torch.no_grad():
            outputs = self.model(**batched_tokens)

        outputs = mean_pooling(outputs, batched_tokens['attention_mask'])
        # Get the embeddings from the model's output
        embeddings = F.normalize(outputs, p=2, dim=1)
        print(embeddings.shape)
        # Reshape the embeddings to 2D
        embeddings = embeddings.numpy()
        print(embeddings.shape)

        # Compute cosine similarity between every pair of sentences
        similarity_matrix = cosine_similarity(embeddings)


        mix_const_1 = []
        mix_const_2 = []
        mix_const_idx_1 = []
        mix_const_idx_2 = []

        # Print pairwise sentence similarities
        for i in range(len(sentences)):
            tar_sentences_idx = [k for k in range(0, len(sentences)) if k != i]
            sim_list = []

            for j in tar_sentences_idx:
                sim_list.append(similarity_matrix[i][j])

            top_5_indexes = sorted(range(len(sim_list)), key=lambda i: sim_list[i], reverse=True)[:2]

            mix_const_1.append(self.df.at[tar_sentences_idx[top_5_indexes[0]],'final_const'])
            mix_const_2.append(self.df.at[tar_sentences_idx[top_5_indexes[1]],'final_const'])
            mix_const_idx_1.append(tar_sentences_idx[top_5_indexes[0]])
            mix_const_idx_2.append(tar_sentences_idx[top_5_indexes[1]])

        self.df["mix_const_1"] = mix_const_1
        self.df["mix_const_2"] = mix_const_2
        self.df["mix_const_idx_1"] = mix_const_idx_1
        self.df["mix_const_idx_2"] = mix_const_idx_2

        self.df.to_csv(os.path.join(self.out_pth,self.out_file+".tsv"),sep="\t",index=False)

        self.save_json_abstract()

    def save_json_abstract(self):
        json_data = []

        for idx, row in self.df.iterrows():
            json_entry = {
                "instruction": row["abs_const"],
                "input": "",
                "output": idx
            }
            json_data.append(json_entry)

        # Write the JSON data to a file
        with open(os.path.join(self.out_pth,self.out_file+"_abs.json"), 'w') as json_file:
            json.dump(json_data, json_file, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-p', "--par_dir", type=str, required=True)
    parser.add_argument('-ds', "--ds_dir", type=str, required=True)
    parser.add_argument('-o', '--out_file', type=str, required=True)
    parser.add_argument('-d', '--debug', type=int, default=False)
    args = parser.parse_args()
    print(f"Main debug value : {args.debug}")
    os.makedirs(args.par_dir, exist_ok=True)
    out_dir = os.path.join(args.par_dir, args.ds_dir)
    os.makedirs(out_dir, exist_ok=True)

    cg = ConstraintGeneration(args.ds_dir, args.input_path, out_dir, args.out_file, args.debug)
    cg.generate_constraints()

if __name__ == "__main__":
    main()