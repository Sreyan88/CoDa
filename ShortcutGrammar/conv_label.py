import pandas as pd
import argparse

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
    "ots": {
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
    "massive": {'alarm_set': 0, 'audio_volume_mute': 1, 'iot_hue_lightchange': 2, 'iot_hue_lightoff': 3, 'iot_hue_lightdim': 4, 'iot_cleaning': 5, 'calendar_query': 6, 'play_music': 7, 'general_quirky': 8, 'general_greet': 9, 'datetime_query': 10, 'datetime_convert': 11, 'takeaway_query': 12, 'alarm_remove': 13, 'alarm_query': 14, 'news_query': 15, 'music_likeness': 16, 'music_query': 17, 'iot_hue_lightup': 18, 'takeaway_order': 19, 'weather_query': 20, 'music_settings': 21, 'general_joke': 22, 'music_dislikeness': 23, 'audio_volume_other': 24, 'iot_coffee': 25, 'audio_volume_up': 26, 'iot_wemo_on': 27, 'iot_hue_lighton': 28, 'iot_wemo_off': 29, 'audio_volume_down': 30, 'qa_stock': 31, 'play_radio': 32, 'recommendation_locations': 33, 'qa_factoid': 34, 'calendar_set': 35, 'play_audiobook': 36, 'play_podcasts': 37, 'social_query': 38, 'transport_query': 39, 'email_sendemail': 40, 'recommendation_movies': 41, 'lists_query': 42, 'play_game': 43, 'transport_ticket': 44, 'recommendation_events': 45, 'email_query': 46, 'transport_traffic': 47, 'cooking_query': 48, 'qa_definition': 49, 'calendar_remove': 50, 'lists_remove': 51, 'cooking_recipe': 52, 'email_querycontact': 53, 'lists_createoradd': 54, 'transport_taxi': 55, 'qa_maths': 56, 'social_post': 57, 'qa_currency': 58, 'email_addcontact': 59}
}

parser = argparse.ArgumentParser(description='A simple script with command-line arguments.')

# Add arguments
parser.add_argument('--dataset', '-d', type=str, help='Path to the input file')
parser.add_argument('--split', '-s', type=str, help='Path to the output file')

# Parse the command-line arguments
args = parser.parse_args()

dataset = args.dataset
split = args.split

df = pd.read_csv(f"./ShortcutGrammar/data/{dataset}/{split}.tsv", sep="\t", header=0)
print(df)
if '0' in df.columns:
    df = df.rename(columns={'0': 'text', '1': 'label'})

print(df)

df["label_name"] = df["label"]

df["label"] = df["label"].apply(lambda x: desc2label[dataset][x])

df.rename(columns={'text': 'sentence'}, inplace=True)

df.to_csv(f"./ShortcutGrammar/data/{dataset}/{split}.tsv", sep="\t", index=False)