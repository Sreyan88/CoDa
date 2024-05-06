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

import random

import spacy
spacy.prefer_gpu()

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

class ConstraintGeneration:
    def __init__(self, ds_name, inp_path, json_path, out_path, out_file, debug):
        self.nlp = spacy.load("en_core_web_lg")
        if debug==1:
            self.df = pd.read_csv(inp_path,sep="\t",header=0).head(20)
        else:
            self.df = pd.read_csv(inp_path,sep="\t",header=0)
        if '0' in self.df.columns:
            self.df = self.df.rename(columns={'0': 'text', '1': 'label'})        
        self.word_regex = re.compile(r'^[A-Za-z]+$')
        self.out_pth = out_path
        self.out_file = out_file
        self.ds_name = ds_name
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = SentenceTransformer(self.model_name)
        # self.abs_con_prompts = self.get_abs_con_prompts(json_path)

    def get_abs_con_prompts(self, json_path):
        abs_prompts = {}
        with open(json_path, "r") as file:
            for line in file:
                obj = json.loads(line)
                concept = obj["predict"].replace('#','').replace(".","").lstrip().rstrip()
                if obj["label"] in abs_prompts:
                    abs_prompts[obj["label"]].append("Any sentence in the document " + "should not include the abstract concept " + concept.lower() + ".")
                else:
                    abs_prompts[obj["label"]] = ["Any sentence in the document " + "should not include the abstract concept " + concept.lower() + "."]               
        print(abs_prompts)
        return abs_prompts
    
    def lemmatize_word(self, word):
        return word
        # lemmatizer = WordNetLemmatizer()
        # # Get the part of speech (pos) for the word
        # pos = nltk.pos_tag([word])[0][1][0].lower()
        
        # # Map the POS tag to WordNet POS tag
        # pos_mapping = {'n': wordnet.NOUN, 'v': wordnet.VERB, 'r': wordnet.ADV, 'a': wordnet.ADJ}
        # wordnet_pos = pos_mapping.get(pos, wordnet.NOUN)
        
        # # Lemmatize the word
        # lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
        
        # return lemmatized_word

    def get_lemma_list(self, word_list):
        res = []
        for word in word_list:
            res.append(self.lemmatize_word(word.lower()))
        return res
        
    def lexical_constraint(self, sentence, noun_list, verb_list, other_list, word_list):
        nv_list = noun_list + verb_list
        random.shuffle(nv_list)
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
        
        nv_final_lem = nv_final

        for idx, _ in enumerate(nv_final):
            if nv_final[idx] in alt_map:
                alt_kw = alt_map[nv_final[idx]]
                if random.randint(0,1) == 1 and (alt_kw not in nv_final_lem):
                    nv_final[idx] = nv_final[idx] + " or " + alt_kw
                if (alt_kw not in nv_final_lem) and (alt_kw not in other_list):
                    alt_kws_list.append(alt_kw)

        print(nv_final)
        print(nv_final_lem)
        print(alt_kws_list)
        print(other_list)

        rand_other_kws = random.sample(other_list, k=random.randint(1, int(max(1, 0.3*len(other_list))))) if len(other_list)>0 else []

        rand_nv_kws = random.sample(nv_final, k=random.randint(1, int(max(1,0.3*len(nv_final))))) if len(nv_final)>0 else []
        
        inc_kws  = (",".join(rand_other_kws) + "," + ",".join(rand_nv_kws)) if len(rand_other_kws) > 0 else ",".join(rand_nv_kws) if len(rand_nv_kws) > 0 else ""
        exc_kws = ""

        if len(alt_kws_list) > 0:
            rand_neg = random.sample(alt_kws_list, k=random.randint(1, int(max(1,0.3*len(alt_kws_list)))))
            exc_kws = ",".join(rand_neg)
            # constraint = f"Write something with the following keywords: {inc_kws}, but do not include the following keywords: {exc_kws}"
            # print(constraint)
            # print()
        # else:
        #     constraint = f"Write something with the following keywords: {inc_kws}"
        inc_kws = inc_kws.rstrip(",")
        exc_kws = exc_kws.rstrip(",")
        return inc_kws, exc_kws

    def get_spacy_data(self, sent):
        doc = self.nlp(sent)
        # keyword_list = []
        pos_list = []
        # noun_list = []
        # verb_list = []
        # other_list = []
        for token in doc:
            pos_list.append(token.pos_)
            # if token.text == "''" or token.text == "'s" or token.text in stopwords.words('english') or len(token.text) < 2 or len(token.text.strip())==0:
            #     continue
            # keyword_list.append(token.text)
            # if token.pos_ == "NOUN":
            #     noun_list.append(token.text)
            # elif token.pos_ == "VERB":
            #     verb_list.append(token.text)
            # else:
            #     other_list.append(token.text)

        # noun_list = list(set(noun_list))
        # verb_list = list(set(verb_list))
        # other_list = list(set(other_list))

        pos_sequence = " ".join(pos_list)

        return pos_sequence#, noun_list, verb_list, other_list

    def get_top_k_kws(self, sent, label):
        words = word_tokenize(sent)
        words = list(set(words))
        words = [word for word in words if not (word == "''" or word == "'s" or word in stopwords.words('english') or len(word) < 2 or len(word.strip())==0)]
        label_list = [label]*len(words)
        # print(label_list)
        label_encodings = self.model.encode(label_list)
        word_encodings = self.model.encode(words)

        cosine_scores = util.cos_sim(word_encodings, label_encodings)

        cos_sim = []

        for i in range(len(words)):
            cos_sim.append(cosine_scores[i][i])

        word_score_pairs = list(zip(words, cos_sim))

        sorted_word_score_pairs = sorted(word_score_pairs, key=lambda x: x[1], reverse=True)

        # print(sorted_word_score_pairs)

        kws = [pair[0] for pair in sorted_word_score_pairs]

        # print(kws)

        return kws[:max(min(3, len(words)), int(0.3 * len(words)))]

    def conc_lexical_constraint(self, sentence, word_list):
        alt_map = {}
        alt_kws_list = []        
        for word in word_list:
            if word in word_list and self.word_regex.match(word):
                alt_list = lexsub_dropout(sentence, word)
            else:
                alt_list = []

            if len(alt_list) > 1 and len(alt_list[0][0]) > 0 and len(alt_list[0][0].strip()) > 1 and self.word_regex.match(alt_list[0][0]):
                alt_map[word] = alt_list[0][0]
        
        for idx, _ in enumerate(word_list):
            if word_list[idx] in alt_map:
                alt_kw = alt_map[word_list[idx]]
                if random.randint(0,1) == 1:
                    word_list[idx] = word_list[idx] + " or " + alt_kw
                elif (alt_kw not in word_list):
                    alt_kws_list.append(alt_kw)

        inc_kws = ",".join(word_list)
        exc_kws = ""

        # print(alt_kws_list)

        if len(alt_kws_list) > 0:
            rand_neg = random.sample(alt_kws_list, k=random.randint(1, int(max(1,0.3*len(alt_kws_list)))))
            exc_kws = ",".join(rand_neg)

        inc_kws = inc_kws.rstrip(",")
        exc_kws = exc_kws.rstrip(",")
        return inc_kws, exc_kws

    def get_exemplars(self, sent_ori, sent, label):
        # print(f"Label: {label}")
        label_df = self.df[self.df["label"] == label]
        # print(sent_ori)
        # print(f"Len of label_df: {len(label_df)}")
        label_sent_list = [x for x in label_df["text"].tolist() if x!=sent_ori]
        # print(label_sent_list)
        rand_sent_list = random.sample(label_sent_list, min(len(label_sent_list), 3))
        random.shuffle(rand_sent_list)
        return ','.join(['"{}"'.format(sentence) for sentence in rand_sent_list])

    def generate_constraints(self):

        topic_dict = {
            "ots": "The document's terms of service should be {label}",
            "yahoo": "The document should be on the topic of {label}",
            "huff": "The document should be on the topic of {label}",
            "atis": "The document should be on the topic of {label}",
            "massive": "The document should be on the topic of {label}"
        }

        topic_dict = topic_dict[self.ds_name]

        prompt_inc_exc =[ 
"""Write a brief document with a single sentence or multiple sentences corresponding to the following abstract description: $abs$.
Additionally, the document should have the following constraints:
1. The document should have the following keywords: {inc_kws}, but should not have the following keywords : {exc_kws}.
2. {topic}. Here are also some examples: {exemplars}.
3. The document should have a length of {min_words}-{max_words} words."""
, 
"""Write a brief document with a single sentence or multiple sentences with the following constraints:
1. The document should have the following keywords: {inc_kws}, but should not have the following keywords : {exc_kws}.
2. {topic}. Here are also some examples: {exemplars}.
3. The document should have a length of {min_words}-{max_words} words."""
        ]

        prompt_inc = [
"""Write a brief document with a single sentence or multiple sentences corresponding to the following abstract description: $abs$.
Additionally, the document should have the following constraints:
1. The document should have the following keywords: {inc_kws}.
2. {topic}. Here are also some examples: {exemplars}.
3. The document should have a length of {min_words}-{max_words} words."""
,
"""Write a brief document with a single sentence or multiple sentences with the following constraints:
1. The document should have the following keywords: {inc_kws}.
2. {topic}. Here are also some examples: {exemplars}.
3. The document should have a length of {min_words}-{max_words} words."""
        ]

        prompt_none = [
"""Write a brief document with a single sentence or multiple sentences corresponding to the following abstract description: $abs$.
Additionally, the document should have the following constraints:
1. {topic}. Here are also some examples: {exemplars}.
2. The document should have a length of {min_words}-{max_words} words."""
,
"""Write a brief document with a single sentence or multiple sentences with the following constraints:
1. {topic}. Here are also some examples: {exemplars}.
2. The document should have a length of {min_words}-{max_words} words."""
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
            sent = clean_pipeline(row["text"])
            print(f"Sent: {sent}")
            abs_prompt = "Write an abstract of the following document in a few words:\n{}".format(sent)
            abs_const_list.append(abs_prompt)

            pos_sequence = self.get_spacy_data(sent)
            pos_seq_list.append(pos_sequence)

            top_k_kws = self.get_top_k_kws(sent, row["label"])

            rand_exemplars= self.get_exemplars(row["text"], sent, row["label"])

            print(f"Random exemplars: {rand_exemplars}")
            print(f"Idx: {idx}, len of top_k_kws: {top_k_kws}")
            print(f"top_k_kws: {top_k_kws}")
            # if idx == 1:
            #     exit()
            # continue
            # pos_seq_list.append(pos_sequence)

            num_words = len(sent.split())
            min_len = max(int(num_words),5)
            max_len = max(int((3*num_words)/2),10)
            min_words_list.append(min_len)
            max_words_list.append(max_len)

            topic = row["label"]
            topic_list.append(topic)

            num_sents = len(sent_tokenize(sent))
            num_sents_list.append(num_sents)

            sent_key = "sentence" if num_sents == 1 else "sentences"
            sent_key_list.append(sent_key)

            try:
                # print("Doing conc_lexical_constraint.")
                inc_kws, exc_kws = self.conc_lexical_constraint(sent, top_k_kws)
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
            elif len(inc_kws) > 0:
                curr_prompt = prompt_inc
                inc = True
            else:
                curr_prompt = prompt_none

            for idx, prompt in enumerate(curr_prompt):
                # start_count = 0
                # if "3." in prompt:
                #     start_count = 4
                # elif "2." in prompt:
                #     start_count = 3
                # else:
                #     start_count = 1
                # for abs_concept in self.abs_con_prompts[row["label"]]:
                #     prompt = prompt + os.linesep + f"{start_count}. " + abs_concept
                #     start_count = start_count + 1
                if idx == 1:
                    if inc_exc:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            # pos_seq=pos_sequence,
                            min_words=min_len,
                            max_words=max_len,
                            topic=topic_dict.format(label=row["label"]),
                            inc_kws=inc_kws,
                            exc_kws=exc_kws
                        )
                        # print(topic_dict.format(label=row["label"]))
                        # print(prompt)
                    elif inc:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            # pos_seq=pos_sequence,
                            min_words=min_len,
                            max_words=max_len,
                            topic=topic_dict.format(label=row["label"]),
                            inc_kws=inc_kws
                        )
                    else:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            # pos_seq=pos_sequence,
                            min_words=min_len,
                            max_words=max_len,
                            topic=topic_dict.format(label=row["label"])
                        )
                    final_const_topic_list.append(prompt) 
                    # print(final_const_topic_list)          
                else:
                    if inc_exc:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            # pos_seq=pos_sequence,
                            min_words=min_len,
                            max_words=max_len,
                            topic=topic_dict.format(label=row["label"]),
                            inc_kws=inc_kws,
                            exc_kws=exc_kws
                        )
                    elif inc:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            # pos_seq=pos_sequence,
                            topic=topic_dict.format(label=row["label"]),
                            min_words=min_len,
                            max_words=max_len,
                            inc_kws=inc_kws
                        )
                    else:
                        prompt = prompt.format(
                            exemplars=rand_exemplars,
                            # pos_seq=pos_sequence,
                            topic=topic_dict.format(label=row["label"]),
                            min_words=min_len,
                            max_words=max_len
                        ) 
                    final_const_list.append(prompt)
                    # print(final_const_list)

        self.df["inc_kws"] = inc_kws_list
        self.df["exc_kws"] = exc_kws_list
        # self.df["pos_seq"] = pos_seq_list
        self.df["topic"] = topic_list
        self.df["min_words"] = min_words_list
        self.df["max_words"] = max_words_list
        self.df["num_sents"] = num_sents_list
        self.df["abs_const"] = abs_const_list

        self.df["final_const"] = final_const_list
        self.df["final_topic_const"] = final_const_topic_list

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        sim_df = self.df[["text"]]

        # Load the pre-trained BERT model and tokenizer
        # model_name = "bert-base-uncased"
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        # List of input sentences
        sentences = sim_df["text"].to_list()

        # Tokenize the sentences and convert them to tensors in batches
        batched_tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)

        # Pass the tokenized batch through the BERT model
        with torch.no_grad():
            outputs = model(**batched_tokens)

        outputs = mean_pooling(outputs, batched_tokens['attention_mask'])
        # Get the embeddings from the model's output
        embeddings = F.normalize(outputs, p=2, dim=1)
        print(embeddings.shape)
        # Reshape the embeddings to 2D
        embeddings = embeddings.numpy()
        print(embeddings.shape)

        # Compute cosine similarity between every pair of sentences
        similarity_matrix = cosine_similarity(embeddings)

        # Print the similarity matrix
        # print("Cosine Similarity Matrix:")
        # print(similarity_matrix)

        mix_const_1 = []
        mix_const_2 = []
        mix_const_3 = []
        mix_const_4 = []
        mix_const_5 = []

        # Print pairwise sentence similarities
        for i in range(len(sentences)):
            tar_sentences_idx = sim_df.index[self.df['label'] == self.df.at[i,'label']].tolist()

            tar_sentences_idx = [sent_idx for sent_idx in tar_sentences_idx if sent_idx!=i]
            # print(f"Idx : {i} tar_sentences_idx: {tar_sentences_idx}")
            if len(tar_sentences_idx) == 0:
                tar_sentences_idx = [i,i]

            if len(tar_sentences_idx) == 1:
                tar_sentences_idx.append(tar_sentences_idx[0])        

            sim_list = []

            for j in tar_sentences_idx:
                sim_list.append(similarity_matrix[i][j])

            top_5_indexes = sorted(range(len(sim_list)), key=lambda i: sim_list[i], reverse=True)[:2]

            mix_const_1.append(self.df.at[tar_sentences_idx[top_5_indexes[0]],'final_const'])
            mix_const_2.append(self.df.at[tar_sentences_idx[top_5_indexes[1]],'final_const'])
            # mix_const_3.append(self.df.at[top_5_indexes[2],'final_const'])
            # mix_const_4.append(self.df.at[top_5_indexes[3],'final_const'])
            # mix_const_5.append(self.df.at[top_5_indexes[4],'final_const'])

        self.df["mix_const_1"] = mix_const_1
        self.df["mix_const_2"] = mix_const_2
        # self.df["mix_const_3"] = mix_const_3
        # self.df["mix_const_4"] = mix_const_4
        # self.df["mix_const_5"] = mix_const_5

        self.df.to_csv(os.path.join(self.out_pth,self.out_file+".tsv"),sep="\t",index=False)

        self.save_json_abstract()

    def save_json_abstract(self):
        json_data = []

        for _, row in self.df.iterrows():
            json_entry = {
                "instruction": row["abs_const"],
                "output": row["label"],
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