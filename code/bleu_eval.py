from nltk.translate.bleu_score import sentence_bleu
import loading_shakespeare as shakes
import english_simple_sent_align as normal_simple
#import make_baseline_translations as make_base

# weights
UNI_WEIGHT = 0.5
BI_WEIGHT = 0.5
TRI_WEIGHT = 0
QUAD_WEIGHT = 0

RESULT_DIR = "./results/"
SHAKES_BASE_RESULT_FNAME = "shakes_baseline_translation_sents.txt"
WIKI_BASE_RESULT_FNAME = "wiki_baseline_translation_sents.txt"
NORMAL_SENTENCE_FNAME = normal_simple.DATA_DIR + normal_simple.SIMPLE_ENGL_DIR + normal_simple.NORMAL_FNAME
SHAKES_DISSIM_RESULT_FNAME = "shakes_dissimilar_translation_sents.txt"
WIKI_DISSIM_RESULT_FNAME = "wiki_dissimilar_translation_sents.txt"
SHAKES_COMBINED_RESULT_FNAME = "shakes_combined_translation_sents.txt"
WIKI_COMBINED_RESULT_FNAME = "wiki_combined_translation_sents.txt"

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
Shakespeare plays into the sentence distance thing:             dissim_auto_shakes_translation
Normal wiki in dissimilar translator:                           
Shakespeare plays into the one with word embeddings:
'''

print("translation type,input data,bleu score")

''' the comparison between original shakespeare
    sentences and hand made translations
'''
manual_shakes_translation = compare_total_bleu_tuples(shakes.get_aligned_sent_tokens())
print(f"manual,shakespeare,{manual_shakes_translation}")

#for i in range(0, 10):
#    print(i)
#    print("Shake og: ", man_aligned_sents[i][0])
#    print("Shake no fear: ", man_aligned_sents[i][1])
#    print(sentence_bleu([man_aligned_sents[i][0]], 
#                        man_aligned_sents[i][1], 
#                        weights=(0.4, 0.4, 0.15, 0.05)))

''' the comparision between original shakespeare 
    sentences and automatic baseline translations
'''
shakes_base_pred_translations = shakes.get_sent_list(RESULT_DIR + SHAKES_BASE_RESULT_FNAME)
shakes_base_pred_align = list(zip(shakes.get_og_sents(), shakes_base_pred_translations))
base_auto_shakes_translation = compare_total_bleu_tuples(shakes.tokenize_sent_pairs(shakes_base_pred_align))
print(f"baseline model,shakespeare,{base_auto_shakes_translation}")


#for i in range(0, 10):
#    print(i)
#    print("Shake og: ", shakes_base_pred_align[i][0])
#    print("Shake trans: ", shakes_base_pred_align[i][1])
#    print(sentence_bleu([shakes_base_pred_align[i][0]], 
#                        shakes_base_pred_align[i][1], 
#                        weights=(0.4, 0.4, 0.15, 0.05)))

''' the comparison between normal wiki sentences
    to hand made simple wiki sentences
'''
wiki_sent_pairs = normal_simple.tokenize_sent_pairs(normal_simple.sent_pairs("test"))
manual_wiki_translation = compare_total_bleu_tuples(wiki_sent_pairs)
print(f"manual,wiki,{manual_wiki_translation}")

''' the comparision between normal wiki sentences and
    baseline automatic simple wiki sentences
'''
wiki_normal_tokens = normal_simple.tokenize_sents(normal_simple.get_sents(normal_simple.get_sent_dict(NORMAL_SENTENCE_FNAME)))
wiki_normal_tokens = [tup[0] for tup in normal_simple.sent_pairs("test")]
base_pred_tokens = shakes.tokenize_sent_list(shakes.get_sent_list(RESULT_DIR + WIKI_BASE_RESULT_FNAME))
base_pred_wiki_translation = compare_total_bleu_lists(wiki_normal_tokens, base_pred_tokens)
print(f"baseline model,wiki,{base_pred_wiki_translation}")

''' the comparision between original shakespeare 
    sentences and automatic dissimilar translations
'''
shakes_dissim_pred_translations = shakes.get_sent_list(RESULT_DIR + SHAKES_DISSIM_RESULT_FNAME)
shakes_dissim_pred_align = list(zip(shakes.get_og_sents(), shakes_dissim_pred_translations))
dissim_auto_shakes_translation = compare_total_bleu_tuples(shakes.tokenize_sent_pairs(shakes_dissim_pred_align))
print(f"dissimilar model,shakespeare,{dissim_auto_shakes_translation}")

''' the comparision between normal wiki sentences and
    dissimilar automatic simple wiki sentences
'''
wiki_normal_tokens = normal_simple.tokenize_sents(normal_simple.get_sents(normal_simple.get_sent_dict(NORMAL_SENTENCE_FNAME)))
wiki_normal_tokens = [tup[0] for tup in normal_simple.sent_pairs("test")]
dissim_pred_tokens = shakes.tokenize_sent_list(shakes.get_sent_list(RESULT_DIR + WIKI_DISSIM_RESULT_FNAME))
dissim_pred_wiki_translation = compare_total_bleu_lists(wiki_normal_tokens, dissim_pred_tokens)
print(f"dissimilar model,wiki,{dissim_pred_wiki_translation}")

''' the comparision between original shakespeare 
    sentences and automatic combined translations
'''
shakes_combined_pred_translations = shakes.get_sent_list(RESULT_DIR + SHAKES_COMBINED_RESULT_FNAME)
shakes_test_sents = shakes.get_sent_list(shakes.DATA_DIR + shakes.ORIGNAL_DIR + shakes.TESTING_PLAY_NAME + shakes.ORIGINAL_INDC + shakes.FILE_END)
shakes_combined_pred_align = list(zip(shakes_test_sents, shakes_combined_pred_translations))
combined_auto_shakes_translation = compare_total_bleu_tuples(shakes.tokenize_sent_pairs(shakes_combined_pred_align))
print(f"combined model,shakespeare,{combined_auto_shakes_translation}")

''' the comparision between normal wiki sentences and
    combined automatic simple wiki sentences
'''
wiki_normal_tokens = normal_simple.tokenize_sents(normal_simple.get_sents(normal_simple.get_sent_dict(NORMAL_SENTENCE_FNAME)))
wiki_normal_tokens = [tup[0] for tup in normal_simple.sent_pairs("test")]
combined_pred_tokens = shakes.tokenize_sent_list(shakes.get_sent_list(RESULT_DIR + WIKI_COMBINED_RESULT_FNAME))
combined_pred_wiki_translation = compare_total_bleu_lists(wiki_normal_tokens, combined_pred_tokens)
print(f"combined model,wiki,{combined_pred_wiki_translation}")