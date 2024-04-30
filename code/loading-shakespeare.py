import random
import math

DATA_DIR = "data/aligned-plays/"
ORIGNAL_DIR = "original/"
MODERN_DIR = "modern/"

ORIGINAL_INDC = "_original"
MODERN_INDC = "_modern"

FILE_END = ".snt.aligned"

TRAIN_PERCENT = .8
NON_TRAIN_PERCENT = .1
RAND_SEED = 0

# names of the play based on the file name
play_name_list = [
    "antony-and-cleopatra",
    "asyoulikeit",
    "errors",
    "hamlet",
    "henryv",
    "juliuscaesar",
    "lear",
    "macbeth",
    "merchant",
    "msnd",
    "muchado",
    "othello",
    "richardiii",
    "romeojuliet",
    "shrew",
    "tempest",
    "twelfthnight"
]

def get_sent_list(f_name) -> list[str]:
    ''' gets the aligned file and converts it to a list of sentences
    '''
    with open(f_name, encoding="utf8") as reader:
        sent_list = []
        for line in reader:
            sent_list.append(line.replace("\n", ""))
        return sent_list

def sent_pairs() -> list[(str, str)]:
    ''' creates sents pairs with the source first and target second
        twelfth night is our testing play
    '''
    original_sent_list = []
    modern_sent_list = []
    for name in play_name_list:
        print("in " + name)
        original_sent_list += get_sent_list(DATA_DIR + ORIGNAL_DIR + name + ORIGINAL_INDC + FILE_END)
        modern_sent_list += get_sent_list(DATA_DIR + MODERN_DIR + name + MODERN_INDC + FILE_END)
    return list(zip(original_sent_list, modern_sent_list))
