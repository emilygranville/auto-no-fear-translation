import csv
import sys

import numpy as np
maxInt = sys.maxsize

# from https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

data_dir = "../data/"
embed_dir = "dictionary-mappings/"

def get_word_embeddings(f_name) -> list[dict]:
    ''' gets the pretrained embeddings from the csv
        and converts it into a list of dictionaries
        with keys "word" and "embeddings"
    '''
    
    embed_list = []

    with open(f_name, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        embed_list = list(reader)

    for item in embed_list:
        embed = item["embeddings"].replace("[", "").replace("]", "").replace(";", "\n")
        new_array = np.fromstring(embed, sep="\n", dtype="float64")
        item["embeddings"] = new_array
        
    return embed_list

embeddings = get_word_embeddings(data_dir + embed_dir + "pretrained-embeddings.csv")
