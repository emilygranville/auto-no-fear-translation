import dissimilar_translation_transformer as dissimilar_trans
import loading_shakespeare as shakes
import english_simple_sent_align as normal_simple

RES_DIR = "./results/"
SHAKES_DISSIM_TRANS_FNAME = "shakes_dissimilar_translation_sents.txt"
WIKI_DISSIM_TRANS_FNAME = "wiki_dissimilar_translation_sents.txt"

def save_tranlated_sents(f_name: str, sentences: list[str]):
    ''' gets the aligned file and converts it to a list of sentences
    '''
    with open(f_name, "w", encoding="utf8") as writer:
        for sent in sentences:
            writer.write(sent + "\n")

def shakespeare_into_dissimilar():
    source_sents = shakes.get_og_sents()
    #transformer, optimizer, next_epoch = dissimilar_trans.make_or_restore_model()
    translations = []
    for sent in source_sents:
        poss_list = dissimilar_trans.run_ten_times(sent)
        new_sent = dissimilar_trans.rerank_and_top(poss_list, sent)
        translations.append(new_sent)

    save_tranlated_sents(RES_DIR + SHAKES_DISSIM_TRANS_FNAME, translations)

def wiki_into_dissimilar():
    wiki_sent_pairs = normal_simple.sent_pairs("test")
    #transformer, optimizer, next_epoch = dissimilar_trans.make_or_restore_model()
    translations = []
    for sent in wiki_sent_pairs:
        poss_list = dissimilar_trans.run_ten_times(sent[0])
        new_sent = dissimilar_trans.rerank_and_top(poss_list, sent[0])
        translations.append(new_sent)
    save_tranlated_sents(RES_DIR + WIKI_DISSIM_TRANS_FNAME, translations)

#shakespeare_into_dissimilar()
wiki_into_dissimilar()