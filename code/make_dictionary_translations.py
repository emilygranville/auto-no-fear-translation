import dictionary_translation_transformer as dict_trans
import loading_shakespeare as shakes
import english_simple_sent_align as normal_simple
import bleu_eval as bleu

RES_DIR = bleu.RESULT_DIR
SHAKES_DICTIONARY_TRANS_FNAME = bleu.SHAKES_DICT_RESULT_FNAME
WIKI_DICTIONARY_TRANS_FNAME = bleu.WIKI_DICT_RESULT_FNAME

def save_tranlated_sents(f_name: str, sentences: list[str]):
    ''' gets the aligned file and converts it to a list of sentences
    '''
    with open(f_name, "w", encoding="utf8") as writer:
        for sent in sentences:
            writer.write(sent + "\n")

def shakespeare_into_dictionary():
    source_sents = shakes.get_og_sents()
    #source_sents = shakes.get_sent_list(shakes.DATA_DIR + shakes.ORIGNAL_DIR + shakes.TESTING_PLAY_NAME + shakes.ORIGINAL_INDC + shakes.FILE_END)
    transformer, optimizer, next_epoch = dict_trans.make_or_restore_model()
    translations = []
    for sent in source_sents:
        new_sent = dict_trans.translate(transformer, sent)
        translations.append(new_sent)

    save_tranlated_sents(RES_DIR + SHAKES_DICTIONARY_TRANS_FNAME, translations)

def wiki_into_dictionary():
    wiki_sent_pairs = normal_simple.sent_pairs("test")
    transformer, optimizer, next_epoch = dict_trans.make_or_restore_model()
    translations = []
    for sent in wiki_sent_pairs:
        new_sent = dict_trans.translate(transformer, sent[0])
        translations.append(new_sent)
    save_tranlated_sents(RES_DIR + WIKI_DICTIONARY_TRANS_FNAME, translations)

shakespeare_into_dictionary()
wiki_into_dictionary()