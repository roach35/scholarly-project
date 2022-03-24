import pandas as pd
from sentence_transformers import SentenceTransformer
import glob
from time import sleep

# Download the BERT model. paraphrase-MiniLM-L6-v2
model = SentenceTransformer("distilbert-base-uncased-finetuned-sst-2-english")

# Read in the data.
tweets = pd.read_pickle("../data/3_9_tweets.pkl")
tweets["embedding"] = pd.NA
tweets = tweets.iloc[0:6000, :]


# BERT, in chunks.
def bert_tweets(tweet):
    """
    BERTify the Tweets.
    
    :param: tweet
    :return: encoded Tweet
    """
    
    # BERTify the chunk.
    return model.encode(str(tweet))


# Set step.
step = 3000

# Loop through DataFrame.
for min_index in range(0, tweets.shape[0], step):
    # Create a sub DataFrame.
    max_index = min_index + step
    sub_data = tweets.iloc[min_index:max_index, :]
    sub_data["embedding"] = sub_data["tweet"].apply(bert_tweets)
    sub_data.to_pickle(f"../data/embedded/bert_{min_index}.pkl")
    print(f"Finished {min_index}-{max_index}")
    sleep(10)

# Create a list of files.
read_files = list()

# Stitch the written files back together.
for file_name in glob.glob("../data/embedded/*.pkl"):
    read_files.append(pd.read_pickle(file_name))

# Concat, write the data.
stitched_data = pd.concat(read_files, axis=0)
stitched_data.to_pickle("../data/3_23_bert_data_test.pkl")
