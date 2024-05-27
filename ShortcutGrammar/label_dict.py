import pandas as pd

# Sample DataFrame
df = pd.read_csv("./diff/data/low_res/100/intent_slot_new/massive_train.tsv", sep="\t", header=0)

# Get unique values from the column
unique_values = df["1"].unique()

# Create a dictionary with indexes
result_dict = {value: index for index, value in enumerate(unique_values)}

print(result_dict)
print()
print(list(result_dict.keys()))
print()
print(len(list(result_dict.keys())))