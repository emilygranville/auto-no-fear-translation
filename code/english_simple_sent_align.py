import csv

data_dir = "../data/"
simple_eng_dir = "simple-english/"

def get_sent_dict(f_name) -> list[dict]:
    ''' gets the aligned file and converts it to a list of dictionaries
        by treating it like an csv file
        the nth line in one is equivalent to the nth line in the other
    '''
    with open(f_name, newline="", encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["article_title","paragraph_number","sentence"], delimiter="\t")
        return list(reader)

normal_sents = get_sent_dict(data_dir + simple_eng_dir + "normal.aligned")
simple_sents = get_sent_dict(data_dir + simple_eng_dir + "simple.aligned")
print(len(normal_sents))
print(len(simple_sents))
combined_sents = list(zip(normal_sents, simple_sents))
print(len(combined_sents))