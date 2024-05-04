import combined_translation_transformer as combined_trans
import loading_shakespeare as shakes
import english_simple_sent_align as normal_simple

RES_DIR = "./results/"
SHAKES_BASE_TRANS_FNAME = "shakes_combined_translation_sents.txt"
WIKI_BASE_TRANS_FNAME = "wiki_combined_translation_sents.txt"

def save_tranlated_sents(f_name: str, sentences: list[str]):
    ''' gets the aligned file and converts it to a list of sentences
    '''
    with open(f_name, "w", encoding="utf8") as writer:
        for sent in sentences:
            writer.write(sent + "\n")

def shakespeare_into_combined():
    #source_sents = shakes.get_og_sents()
    source_sents = shakes.get_sent_list(shakes.DATA_DIR + shakes.ORIGNAL_DIR + shakes.TESTING_PLAY_NAME + shakes.ORIGINAL_INDC + shakes.FILE_END)
    transformer, optimizer, next_epoch = combined_trans.make_or_restore_model()
    translations = []
    for sent in source_sents:
        new_sent = combined_trans.translate(transformer, sent)
        translations.append(new_sent)

    save_tranlated_sents(RES_DIR + SHAKES_BASE_TRANS_FNAME, translations)

def wiki_into_combined():
    wiki_sent_pairs = normal_simple.sent_pairs("test")
    transformer, optimizer, next_epoch = combined_trans.make_or_restore_model()
    translations = []
    for sent in wiki_sent_pairs:
        new_sent = combined_trans.translate(transformer, sent[0])
        translations.append(new_sent)
    save_tranlated_sents(RES_DIR + WIKI_BASE_TRANS_FNAME, translations)
