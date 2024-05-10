import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def getImageName(folder):
    images_names = []
    for file_name in os.listdir(folder):
        if file_name.endswith('jpg'):
            images_names.append(file_name)
    return images_names


real_Images = getImageName("data/real")
ai_Images = getImageName("data/ai-midjourney")

df_real = pd.DataFrame({"Name": real_Images, "label": "Real"})
df_ai = pd.DataFrame({"Name": ai_Images, "label": "AI"})

df = pd.concat([df_real, df_ai])
df.to_csv("data/Data.csv", index=False)


df = pd.concat([df_real, df_ai]).sample(frac=1, random_state=42)

# Split the data into 90% for training and 10% for testing
df_train, df_test = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

# Split the remaining 90% of the data into 80% for training and 20% for validation
df_train, df_valid = train_test_split(df_train, test_size=0.2, stratify=df_train['label'], random_state=42)

# Create directories for train, valid, and test sets if they don't exist
os.makedirs("data/Train", exist_ok=True)
os.makedirs("data/Valid", exist_ok=True)
os.makedirs("data/Test", exist_ok=True)

# Save the DataFrames to CSV files
df_train.to_csv("data/Train/Train.csv", index=False)
df_valid.to_csv("data/Valid/Validation.csv", index=False)
df_test.to_csv("data/Test/Test.csv", index=False)