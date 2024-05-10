import pandas as pd


def encoding(dir):
    df = pd.read_csv(dir)
    df['label'] = df['label'].map({'Real': 0, 'AI': 1})
    df.to_csv(dir, index=False)


encoding("data/Train/Train.csv")
encoding("data/Valid/Validation.csv")
encoding("data/Test/Test.csv")
