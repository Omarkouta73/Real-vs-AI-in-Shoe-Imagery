import os
import shutil
import pandas as pd

train_df = pd.read_csv("data/Train/Train.csv")
valid_df = pd.read_csv("data/Valid/Validation.csv")
test_df = pd.read_csv("data/Test/Test.csv")


def move_images(df, dir):
    for index, row in df.iterrows():
        image_name = row['Name']
        # Define the source path of the image
        if row['label'] == "AI":
            source_path = os.path.join("data/ai-midjourney", image_name)
        elif row['label'] == "Real":
            source_path = os.path.join("data/real", image_name)

        destination_path = os.path.join(dir, image_name)
        shutil.move(source_path, destination_path)


move_images(train_df, "data/Train")
move_images(valid_df, "data/Valid")
move_images(test_df, "data/Test")
