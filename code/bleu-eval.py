from nltk.translate.bleu_score import sentence_bleu
import loading_shakespeare as shakes
import english_simple_sent_align as normal_simple
import make_baseline_translations as make_base

# weights
UNI_WEIGHT = 0.25
BI_WEIGHT = 0.25
TRI_WEIGHT = 0
QUAD_WEIGHT = 0

RESULT_DIR = make_base.RES_DIR

SHAKES_BASE_RESULT_FNAME = make_base.SHAKES_BASE_TRANS_FNAME
WIKI_BASE_RESULT_FNAME = make_base.WIKI_BASE_TRANS_FNAME
NORMAL_SENTENCE_FNAME = normal_simple.DATA_DIR + normal_simple.SIMPLE_ENGL_DIR + normal_simple.NORMAL_FNAME

def compare_total_bleu_lists(ref: list[list[str]], pred: list[list[str]]) -> float:
    ''' gets average bleu score for the sentences when they are two
        separate lists of tokens
    '''
    ref_len = len(ref)
    if not ref_len == len(pred):
        return -1
    
    total = 0
    weights = (UNI_WEIGHT, BI_WEIGHT, TRI_WEIGHT, QUAD_WEIGHT)
    for i in range(0, ref_len):
        total += sentence_bleu([ref[i]], pred[i], weights=weights)
    
    return total / ref_len

def compare_total_bleu_tuples(combined: list[tuple[list[str], list[str]]]) -> float:
    ''' gets the average bleu score for the sentence when they are
        in a list of tuples with the reference first and prediction
        second, already tokenized
    '''
    total_len = len(combined)
    total = 0
    weights = (UNI_WEIGHT, BI_WEIGHT, TRI_WEIGHT, QUAD_WEIGHT)
    for tup in combined:
        total += sentence_bleu([tup[0]], tup[1], weights=weights)

    return total/total_len


'''
Shakespeare to real no fear:                                    manual_shakes_translation
Actual wiki aligned sentences:                                  manual_wiki_translation
Normal wiki into baseline translator:                           base_pred_wiki_translation
Shakespeare plays into the baseline:                            base_auto_shakes_translation
Shakespeare plays into the sentence distance thing:
Shakespeare plays into the one with word embeddings:
'''

''' the comparison between original shakespeare
    sentences and hand made translations
'''
manual_shakes_translation = compare_total_bleu_tuples(shakes.get_aligned_sent_tokens())
print(f"manual shakespeare translation score,{manual_shakes_translation}")

''' the comparision between original shakespeare 
    sentences and automatic baseline translations
'''
shakes_base_pred_translations = shakes.get_sent_list(RESULT_DIR + SHAKES_BASE_RESULT_FNAME)
shakes_base_pred_align = list(zip(shakes.get_og_sents(), shakes_base_pred_translations))
base_auto_shakes_translation = compare_total_bleu_tuples(shakes.tokenize_sent_pairs(shakes_base_pred_align))
print(f"baseline shakespeare translation prediction score,{base_auto_shakes_translation}")

''' the comparison between normal wiki sentences
    to hand made simple wiki sentences
'''
wiki_sent_pairs = normal_simple.tokenize_sent_pairs(normal_simple.sent_pairs("test"))
manual_wiki_translation = compare_total_bleu_tuples(wiki_sent_pairs)
print(f"manual wiki translation score,{manual_wiki_translation}")

''' the comparision between normal wiki sentences and
    automatic simple wiki sentences
'''
wiki_normal_tokens = normal_simple.tokenize_sents(normal_simple.get_sents(normal_simple.get_sent_dict(NORMAL_SENTENCE_FNAME)))
wiki_normal_tokens = [tup[0] for tup in normal_simple.sent_pairs("test")]
base_pred_tokens = shakes.tokenize_sent_list(shakes.get_sent_list(RESULT_DIR + WIKI_BASE_RESULT_FNAME))
base_pred_wiki_translation = compare_total_bleu_lists(wiki_normal_tokens, base_pred_tokens)
print(f"baseline wiki translation prediction score,{base_pred_wiki_translation}")