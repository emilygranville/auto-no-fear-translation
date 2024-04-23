import csv
import sys
import pandas as pd

import numpy as np
maxInt = sys.maxsize

data_dir = "data/"
embed_dir = "dictionary-mappings/short-embeddings/"

# from https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

EMBED_FILE_NAME_LIST = [
    "short-embed01.csv", "short-embed02.csv", "short-embed03.csv",
    "short-embed04.csv", "short-embed05.csv", "short-embed06.csv",
    "short-embed07.csv", "short-embed08.csv", "short-embed09.csv",
    "short-embed10.csv", "short-embed11.csv", "short-embed12.csv",
    "short-embed13.csv", "short-embed14.csv", "short-embed15.csv",
    "short-embed16.csv", "short-embed17.csv", "short-embed18.csv",
    "short-embed19.csv", "short-embed20.csv", "short-embed21.csv",
    "short-embed22.csv",
]

small_list = [
    "short-embed04.csv"
]

def get_word_embeddings(dir_name) -> dict:
    ''' gets the pretrained embeddings from the csv
        and converts it into a dictionary
        with the words as keys and the embeddings
        as the value
    '''

    total_dictionary = {}

    #for name in small_list:
    for name in EMBED_FILE_NAME_LIST:
        embed_list = []

        with open((dir_name + name), newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")
            embed_list = list(reader)

        for item in embed_list:
            try:
                embed = item["embeddings"].replace("[", "").replace("]", "").replace(";", "\n")
                new_array = np.fromstring(embed, sep="\n", dtype="float64")
                total_dictionary[item["word"]] = new_array
            except:
                print("error in" + name + " item: " + str(item)[:20])
        
    return total_dictionary

embeddings = get_word_embeddings(data_dir + embed_dir)
print("length: "+str(len(embeddings.keys())))