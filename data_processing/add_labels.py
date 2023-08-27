import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import DatasetDict, Dataset
from scipy.special import softmax


# Read in the data
def read_data(file_path):
    return pd.read_csv(file_path)


class CFG:
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    file_path = "../data/cleaned_tweets.csv"
    batch_size = 64
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    df = read_data(file_path)


def add_labels():
    df = CFG.df
    tokenizer = CFG.tokenizer

    # Get all the tweets features into  series
    tweet_features = df[["lemmatized_text"]]

    tweet_features = DatasetDict({"df_tweets": Dataset.from_pandas(tweet_features)})

    encoded_tweets_batches = []
    num_samples = len(tweet_features["df_tweets"]["lemmatized_text"])

    for i in range(0, num_samples, CFG.batch_size):
        batch = tweet_features["df_tweets"]["lemmatized_text"][i:i+CFG.batch_size]

        encoded_batch = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=CFG.config.max_position_embeddings,
            return_tensors="pt"
        )
        encoded_tweets_batches.append(encoded_batch)

    # Get the labels
    labels = []
    for encoded_batch in encoded_tweets_batches:
        logits = CFG.model(**encoded_batch)[0].detach().numpy()
        batch_labels = [softmax(x) for x in logits]
        labels.extend(batch_labels)

    # Add the labels to the dataframe
    df["labels"] = labels
    df.to_csv("../data/labeled_tweets.csv", index=False)
    print(f"Labels added to the dataframe (labels in order: {CFG.config.id2label}): \n{df.head(2)}")


if __name__ == "__main__":
    add_labels()
