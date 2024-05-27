import os
import tiktoken
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Union, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, GenerationConfig, InfNanRemoveLogitsProcessor, LogitsProcessorList
from torch.utils.data import DataLoader
from huggingface_hub.hf_api import HfFolder
import datasets
from datasets import load_dataset, Dataset
import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
parser.add_argument('--model', '-m', required=True, type=str)
parser.add_argument('--config_name', '-c', required=True, type=str)
parser.add_argument('--num_return_sequences', '-nr', required=True, type=int)
args = parser.parse_args()

desc2label = {
    "yahoo":{
        "Society & Culture": 0,
        "Science & Mathematics": 1,
        "Health": 2,
        "Education & Reference": 3,
        "Computers & Internet": 4,
        "Sports": 5,
        "Business & Finance": 6,
        "Entertainment & Music": 7,
        "Family & Relationships": 8,
        "Politics & Government": 9
    },
    "sst2":{
        "positive": 1,
        "negative": 0        
    },
    "ots":{
        "potentially unfair": 0,
        "clearly unfair": 1,
        "clearly fair": 2
    }
}

def get_lists_from_jsonl(file, multiple):
    with open(file,'r') as f:
        data = json.load(f)
    prompt_list = []
    label_list = []
    for i in range(multiple):
        for obj in data:
            prompt_list.append(obj["instruction"])
            if len(obj["input"]) > 0:
                label_list.append(obj["input"])
            else:
                label_list.append(obj["output"])
    return prompt_list, label_list

def get_raw_dataset():
    prompt_list= []
    label_list = []
    
    file_path = f'./generation_data/{args.config_name}.json'
    prompt_list, label_list = get_lists_from_jsonl(file_path, 1)

    raw_datasets = datasets.DatasetDict({'text': prompt_list, 'label': label_list, 'timestamp': [0]*len(prompt_list), 'url':len(prompt_list)*[0]})
    raw_datasets = Dataset.from_dict(raw_datasets)
    raw_datasets = datasets.DatasetDict({'train': raw_datasets})
    return raw_datasets

@dataclass
class Template:

    prefix: List[Union[str, Dict[str, str]]]
    prompt: List[Union[str, Dict[str, str]]]
    system: str
    sep: List[Union[str, Dict[str, str]]]
    stop_words: List[str]
    use_history: bool
    efficient_eos: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids = prompt_ids + query_ids + resp_ids
        prompt_ids, answer_ids = prompt_ids + encoded_pairs[-1][0], encoded_pairs[-1][1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        return encoded_pairs

    def _format(
        self,
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[str, List[Tuple[str, str]]]:
        r"""
        Aligns inputs to the standard format.
        """
        system = system or self.system # use system if provided
        history = []
        history = history + [(query, resp)]
        return system, history

    def _get_special_ids(
        self,
        tokenizer: "PreTrainedTokenizer"
    ) -> Tuple[List[int], List[int]]:
        if tokenizer.bos_token_id is not None and getattr(tokenizer, "add_bos_token", True):
            bos_ids = [tokenizer.bos_token_id]
        else: # baichuan, qwen and gpt2 models have no bos token
            bos_ids = []

        if tokenizer.eos_token_id is None:
            raise ValueError("EOS token is required.")

        if self.efficient_eos: # used in baichuan, qwen, chatglm, etc.
            eos_ids = []
        else:
            eos_ids = [tokenizer.eos_token_id]

        return bos_ids, eos_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        query: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + sep + query    resp + eos
        Turn t: sep + bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        sep_ids = self._convert_inputs_to_ids(tokenizer, context=self.sep)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0:
                prefix_ids = self._convert_inputs_to_ids(tokenizer, context=self.prefix, system=system)
                if len(prefix_ids) != 0: # has prefix
                    prefix_ids = bos_ids + prefix_ids + sep_ids
                else:
                    prefix_ids = bos_ids
            else:
                prefix_ids = sep_ids + bos_ids

            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query, idx=str(turn_idx))
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((prefix_ids + query_ids))
            
        return encoded_pairs

    def _convert_inputs_to_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        context: List[Union[str, Dict[str, str]]],
        system: Optional[str] = None,
        query: Optional[str] = None,
        idx: Optional[str] = None
    ) -> List[int]:
        r"""
        Converts context to token ids.
        """
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        token_ids = []
        for elem in context:
            if isinstance(elem, str):
                elem = elem.replace("{{system}}", system, 1) if system is not None else elem
                elem = elem.replace("{{query}}", query, 1) if query is not None else elem
                elem = elem.replace("{{idx}}", idx, 1) if idx is not None else elem
                if len(elem) != 0:
                    token_ids = token_ids + tokenizer.encode(elem, **kwargs)
            elif isinstance(elem, dict):
                token_ids = token_ids + [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            else:
                raise ValueError("Input must be string or dict[str, str], got {}".format(type(elem)))

        return token_ids

class Llama2Template(Template):

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + query    resp + eos
        Turn t: bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0: # llama2 template has no sep_ids
                query = self.prefix[0].replace("{{system}}", system) + query
            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query)
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((bos_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs

templates: Dict[str, Template] = {}

def register_template(
    name: str,
    prefix: List[Union[str, Dict[str, str]]],
    prompt: List[Union[str, Dict[str, str]]],
    system: str,
    sep: List[Union[str, Dict[str, str]]],
    stop_words: Optional[List[str]] = [],
    use_history: Optional[bool] = True,
    efficient_eos: Optional[bool] = False
) -> None:
    template_class = Llama2Template
    templates[name] = template_class(
        prefix=prefix,
        prompt=prompt,
        system=system,
        sep=sep,
        stop_words=stop_words,
        use_history=False,
        efficient_eos=efficient_eos
    )

register_template(
    name="llama2",
    prefix=[
        "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST]"
    ],
    system=(
        "You are an assistant that only returns what is requested for and nothing else."
        "If you are questioned to return a document, only return the document and not a single extra word."
        "You will be asked to generate a document following some constraints, and you will only output documents that follow all constraints."
    ),
    sep=[]
)

def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:
    
    # if tokenizer.eos_token_id is None:
    #     tokenizer.eos_token = "<|endoftext|>"
    #     # logger.info("Add eos token: {}".format(tokenizer.eos_token))

    # if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
        # logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if name is None:
        return None

    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=template.stop_words),
        replace_additional_special_tokens=False
    )
    return template

model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, access_token="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", padding_side='left')

raw_datasets = get_raw_dataset()

# raw_datasets['train'] = raw_datasets['train'].select(range(92,100))

print(len(raw_datasets['train']))

print(raw_datasets["train"]["text"][0])

template = get_template_and_fix_tokenizer("llama2", tokenizer)

kwargs = dict(num_proc=4, desc="Running tokenizer on dataset")

def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
    for i in range(len(examples['text'])):
        query = examples["text"][i]
        yield query

def preprocess_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
    model_inputs = {"input_ids": [], "attention_mask": []}
    for query in construct_example(examples):
        input_ids, _ = template.encode_oneturn(tokenizer, query, "")
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
    return model_inputs

dataset = raw_datasets.map(preprocess_dataset, batched=True, remove_columns=['text', 'label', 'timestamp', 'url'], **kwargs)

print(dataset)

data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=None,
        label_pad_token_id=tokenizer.pad_token_id
)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_safetensors = False)

generating_args = {}
generating_args.update(dict(
    do_sample=True,
    temperature=0.5,
    top_k=50,
    num_return_sequences=args.num_return_sequences,
    eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
    pad_token_id=tokenizer.pad_token_id,
    max_length = 2000
))

def get_logits_processor() -> LogitsProcessorList:
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor

dataloader = DataLoader(dataset['train'], batch_size=2, collate_fn=data_collator)

label_list_1 = list(raw_datasets["train"]["label"])

label_list = []
for label in label_list_1:
    for i in range (args.num_return_sequences):
        label_list.append(label)

count = 0

new_rows = []

output_path = f"./generation_data/{args.config_name}/"

os.makedirs(output_path, exist_ok=True)

output_prediction_file = os.path.join(output_path, "generated_predictions.jsonl")

file = open(output_prediction_file, "w", encoding="utf-8")
file.close()

for data in tqdm(dataloader):
    
    gen_kwargs = dict(
            generation_config=GenerationConfig(**generating_args),
            logits_processor=get_logits_processor()
        )
    generate_output = model.generate(input_ids = data['input_ids'].cuda(), attention_mask = data['attention_mask'].cuda(),**gen_kwargs)
    response = tokenizer.batch_decode(generate_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print(f"Len of response : {len(response)}")

    with open(output_prediction_file, "a", encoding="utf-8") as writer:
        for i in response:
            temp_prompt = i.split('[/INST]')[0]
            temp_res = i.split('[/INST]')[1]
            print(temp_res)
            write_text = f'{temp_res}'.encode('ascii', 'ignore').decode('ascii')
            writer.write(json.dumps({"label": label_list[count], "predict": temp_res}))
            writer.write("\n")
            count = count + 1