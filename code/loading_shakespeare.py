from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

DATA_DIR = "data/aligned-plays/"
ORIGNAL_DIR = "original/"
MODERN_DIR = "modern/"

ORIGINAL_INDC = "_original"
MODERN_INDC = "_modern"

FILE_END = ".snt.aligned"

TRAIN_PERCENT = .8
NON_TRAIN_PERCENT = .1
RAND_SEED = 0

nlp = English()
tokenizer = Tokenizer(nlp.vocab)

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

def get_og_sents() -> list[str]:
    ''' gets a list of all the original sentences
    '''
    original_sent_list = []
    for name in play_name_list:
        original_sent_list += get_sent_list(DATA_DIR + ORIGNAL_DIR + name + ORIGINAL_INDC + FILE_END)
    return original_sent_list

def sent_pairs() -> list[tuple[str, str]]:
    ''' creates sents pairs with the source first and target second
        twelfth night is our testing play
    '''
    original_sent_list = []
    modern_sent_list = []
    for name in play_name_list:
        original_sent_list += get_sent_list(DATA_DIR + ORIGNAL_DIR + name + ORIGINAL_INDC + FILE_END)
        modern_sent_list += get_sent_list(DATA_DIR + MODERN_DIR + name + MODERN_INDC + FILE_END)
    return list(zip(original_sent_list, modern_sent_list))

def tokenize_sents(combined_sents: list[tuple[str, str]]) -> list[tuple[list[str], list[str]]]:
    ''' creates tokens for the aligned sentences
        returns a list of tuples where the first item is the list
        of tokens in the source sentence and the second item is the
        list of tokens in the target sentence--tokens are converted
        to strings
    '''
    token_tuple_list = []
    for line in combined_sents:
        og_sent_tokens = []
        for token in tokenizer(line[0]):
            og_sent_tokens.append(token.text)
        
        mod_sent_tokens = []
        for token in tokenizer(line[0]):
            mod_sent_tokens.append(token.text)
        
        token_tuple_list.append((og_sent_tokens, mod_sent_tokens))
    return token_tuple_list