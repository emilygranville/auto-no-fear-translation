import translation_transformer as base_trans
import loading_shakespeare as shakes

RES_DIR = "./results/"
BASE_TRANS_FNAME = "baseline_translation_sents.txt"

def save_tranlated_sents(f_name: str, sentences: list[str]):
    ''' gets the aligned file and converts it to a list of sentences
    '''
    with open(f_name, "w", encoding="utf8") as writer:
        for sent in sentences:
            writer.write(sent + "\n")

source_sents = shakes.get_og_sents()
transformer, optimizer, next_epoch = base_trans.make_or_restore_model()
translations = []
for sent in source_sents[0:10]:
    new_sent = base_trans.translate(transformer, sent)
    translations.append(new_sent)

save_tranlated_sents(RES_DIR + BASE_TRANS_FNAME, translations)