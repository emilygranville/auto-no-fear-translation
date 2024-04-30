import csv
import random
import math

DATA_DIR = "data/"
SIMPLE_ENGL_DIR = "simple-english/"

TRAIN_PERCENT = .8
NON_TRAIN_PERCENT = .1
RAND_SEED = 0

def get_sent_dict(f_name) -> list[dict]:
    ''' gets the aligned file and converts it to a list of dictionaries
        by treating it like an csv file
        the nth line in one is equivalent to the nth line in the other
    '''
    with open(f_name, newline="", encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["article_title","paragraph_number","sentence"], delimiter="\t")
        return list(reader)

def get_sents(dictionary: list[dict]) -> list[str]:
    ''' creates a list of just the sentences from the dictionary list
    '''
    sent_list = []
    for item in dictionary:
        sent_list.append(item["sentence"])
    return sent_list

def get_sent_tuples(combined_sent: list[dict]) -> list[tuple]:
    ''' given the result of get_sent_dict, creates tuples of the
        normal (source) sents and the simple (target) sents
    '''
    new_list = []
    for item in combined_sent:
        normal = item[0]["sentence"]
        simple = item[1]["sentence"]
        new_list.append((normal, simple))
    return new_list

def tokenize_sents(sents: list[str]) -> list[list[str]]:
    ''' given a list of sentences, returns a list of token lists
    '''
    list_sent = []
    for sent in sents:
        list_sent.append(sent.split(" "))
    return list_sent

def create_giant_token_list(token_list_list: list[list[str]]) -> list[str]:
    ''' creates a giant list of all the tokens in the entire set of sentences
    '''
    total_list = []
    for sent in token_list_list:
        for token in sent:
            total_list.append(token)
    return total_list

def give_tokens(lang: str) -> list[str]:
    '''creates the tokens in a way that the transformer can use it
    '''
    if lang == "en":
        normal_sents = get_sent_dict(DATA_DIR + SIMPLE_ENGL_DIR + "normal.aligned")
        normal_tokens = tokenize_sents(get_sents(normal_sents))
        all_normal_tokens = create_giant_token_list(normal_tokens)
        return all_normal_tokens
    elif lang == "simple":
        simple_sents = get_sent_dict(DATA_DIR + SIMPLE_ENGL_DIR + "simple.aligned")
        simple_tokens = tokenize_sents(get_sents(simple_sents))
        all_simple_tokens = create_giant_token_list(simple_tokens)
        return all_simple_tokens
    else:
        print("Error")

def sent_pairs(split: str) -> list[(str, str)]:
    ''' creates sents pairs with the source first and target second
    '''
    normal_sents = get_sents(get_sent_dict(DATA_DIR + SIMPLE_ENGL_DIR + "normal.aligned"))
    simple_sents = get_sents(get_sent_dict(DATA_DIR + SIMPLE_ENGL_DIR + "simple.aligned"))
    entire_list = list(zip(normal_sents, simple_sents))
    random.Random(RAND_SEED).shuffle(entire_list)
    
    train_len = math.floor(len(entire_list) * TRAIN_PERCENT)
    non_train_len = train_len + math.floor(len(entire_list) * NON_TRAIN_PERCENT)

    if split == "train":
        return entire_list[:train_len]
    elif split == "valid":
        return entire_list[train_len:non_train_len]
    else:
        return entire_list[non_train_len:]