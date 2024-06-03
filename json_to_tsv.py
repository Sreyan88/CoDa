import pandas as pd
import json
import re
import argparse

parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')
parser.add_argument('--dataset', '-d', type=str, help='Path to the input file')
parser.add_argument('--split', '-s', type=str, help='Path to the output file')
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
    "imdb":{
        "positive": 1,
        "negative": 0        
    },
    "ots":{
        "potentially unfair": 0,
        "clearly unfair": 1,
        "clearly fair": 2
    },
    "huff": {
        'WELLNESS': 0, 
        'POLITICS': 1, 
        'TRAVEL': 2, 
        'ENTERTAINMENT': 3, 
        'STYLE & BEAUTY': 4
    },
    "atis": {'flight': 0, 'flight_time': 1, 'airfare': 2, 'aircraft': 3, 'ground_service': 4, 'airport': 5, 'airline': 6, 'distance': 7, 'abbreviation': 8, 'ground_fare': 9, 'quantity': 10, 'city': 11, 'flight_no': 12, 'capacity': 13, 'flight+airfare': 14, 'meal': 15, 'restriction': 16},
    "massive": {'alarm_set': 0, 'audio_volume_mute': 1, 'iot_hue_lightchange': 2, 'iot_hue_lightoff': 3, 'iot_hue_lighton': 4, 'iot_hue_lightdim': 5, 'iot_cleaning': 6, 'calendar_query': 7, 'play_music': 8, 'general_quirky': 9, 'general_greet': 10, 'datetime_query': 11, 'datetime_convert': 12, 'takeaway_query': 13, 'alarm_remove': 14, 'alarm_query': 15, 'news_query': 16, 'music_likeness': 17, 'music_query': 18, 'iot_hue_lightup': 19, 'takeaway_order': 20, 'weather_query': 21, 'music_settings': 22, 'audio_volume_down': 23, 'general_joke': 24, 'music_dislikeness': 25, 'audio_volume_other': 26, 'iot_coffee': 27, 'audio_volume_up': 28, 'iot_wemo_on': 29, 'iot_wemo_off': 30, 'qa_stock': 31, 'play_radio': 32, 'social_post': 33, 'recommendation_locations': 34, 'cooking_recipe': 35, 'qa_factoid': 36, 'recommendation_events': 37, 'calendar_set': 38, 'play_audiobook': 39, 'play_podcasts': 40, 'social_query': 41, 'transport_query': 42, 'email_sendemail': 43, 'transport_ticket': 44, 'recommendation_movies': 45, 'lists_query': 46, 'play_game': 47, 'email_query': 48, 'transport_traffic': 49, 'cooking_query': 50, 'qa_definition': 51, 'calendar_remove': 52, 'lists_remove': 53, 'email_querycontact': 54, 'lists_createoradd': 55, 'email_addcontact': 56, 'transport_taxi': 57, 'qa_maths': 58, 'qa_currency': 59}
}

df = pd.read_csv(f"./tsv_data/inp_data/{args.dataset}_{args.split}.tsv",sep="\t",header=0)

if '0' in df.columns:
    df = df.rename(columns={'0': 'text', '1': 'label'})

# Specify the file path
file_path = f'./generation_data/{args.dataset}_{args.split}_abs_prompt_{{}}/generated_predictions.jsonl'
solo_file_path = f'./generation_data/{args.dataset}_{args.split}_solo_constraint/generated_predictions.jsonl'

new_rows = []

desc2label = desc2label[args.dataset]

# for k in [1, 2, 3]:
    # Open and read the JSON file
with open(solo_file_path, 'r') as file:
    i = 0
    for line in file:
        obj = json.loads(line)
        predict_split = obj["predict"].split(":")
        if len(predict_split) > 1:
            predict_split = predict_split[1].split("\n\n")
            for text in predict_split:
                if len(text.strip())>0:
                    predict_split = text
                    break
        else:
            predict_split = obj["predict"]       
        if len(str(predict_split))>0:
            new_rows.append({'text': str(predict_split).replace('"','').replace("\n","").replace("\n\n","").replace("\n\n\n","").strip('"').strip('"').strip('"').strip(), 'label':obj["label"]})
        i = i + 1

for k in [1, 2]:
    with open(file_path.format(k), 'r') as file:
        i = 0
        lines = []
        count  = 0
        for line in file:
            obj = json.loads(line)
            predict_split = obj["predict"].split(":")
            if len(predict_split) > 1:
                predict_split = predict_split[1].split("\n\n")
                for text in predict_split:
                    if len(text.strip())>0:
                        predict_split = text
                        break
            else:
                predict_split = obj["predict"]       
            if len(str(predict_split))>0:
                new_rows.append({'text': str(predict_split).replace('"','').replace("\n","").replace("\n\n","").replace("\n\n\n","").strip('"').strip('"').strip('"').strip(), 'label':obj["label"]})
            i = i+1

df = df[["text", "label"]]

df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

df["label"] = df["label"].apply(lambda x : desc2label[x])

df.to_csv(f"./tsv_data/out_data/{args.dataset}/{args.dataset}_{args.split}_ori+aug.tsv",sep="\t",index=False)
