import os
import shutil
import pandas as pd

# Load the train.csv file
train_df = pd.read_csv("data/Train/Train.csv")
valid_df = pd.read_csv("data/Valid/Validation.csv")
test_df = pd.read_csv("data/Test/Test.csv")

def move_images(df, dir):
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Get the image name from the 'Name' column
        image_name = row['Name']
        # Define the source path of the image
        if row['label'] == "AI":
            source_path = os.path.join("data/ai-midjourney", image_name)
        elif row['label'] == "Real":
            source_path = os.path.join("data/real", image_name)
        # Define the destination path where the image will be moved
        destination_path = os.path.join(dir, image_name)
        # Move the image to the train folder
        shutil.move(source_path, destination_path)


move_images(train_df, "data/Train")
move_images(valid_df, "data/Valid")
move_images(test_df, "data/Test")