import pandas as pd

# Assuming your file is named 'your_file.tsv'
df = pd.read_csv('./tsv_data/inp_data/yahoo_val.tsv', delimiter='\t', header=0)

# Identify unique labels
unique_labels = df['label'].unique()

# Create a sampled DataFrame
sampled_rows = pd.DataFrame()

for label in unique_labels:
    label_subset = df[df['label'] == label]
    sampled_rows = pd.concat([sampled_rows, label_subset.sample(frac=500/len(df), replace=True)])

# Shuffle the DataFrame
sampled_rows = sampled_rows.sample(frac=1).reset_index(drop=True)

# Save the sampled data
sampled_rows.to_csv('./tsv_data/inp_data/yahoo_val.tsv', sep='\t', index=False)