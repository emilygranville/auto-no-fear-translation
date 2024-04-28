import csv

data_dir = "data/"
simple_eng_dir = "simple-english/"

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

def get_sent_tuples (combined_sent: list[dict]) -> list[tuple]:
    ''' given the result of get_sent_dict, creates tuples of the
        normal (source) sents and the simple (target) sents
    '''
    new_list = []
    for item in combined_sent:
        normal = item[0]["sentence"]
        simple = item[1]["sentence"]
        new_list.append((normal, simple))
    return new_list

def tokenize_sents (sents: list[str]) -> list[list[str]]:
    ''' given a list of sentences, returns a list of token lists
    '''
    list_sent = []
    for sent in sents:
        list_sent.append(sent.split(" "))
    return list_sent

normal_sents = get_sent_dict(data_dir + simple_eng_dir + "normal.aligned")
simple_sents = get_sent_dict(data_dir + simple_eng_dir + "simple.aligned")

#combined_sents = list(zip(normal_sents, simple_sents))

normal_tokens = tokenize_sents(get_sents(normal_sents))
simple_tokens = tokenize_sents(get_sents(simple_sents))
combined_tokens = list(zip(normal_tokens, simple_tokens))