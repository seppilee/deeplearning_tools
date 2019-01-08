"""
conservative: standard OpenNMT tokenization
aggressive: standard OpenNMT tokenization but only keep sequences of letters/numbers (e.g. splits "2,000" to "2 , 0000", "soft-landing" to "soft - landing")
char: character tokenization
space: space tokenization
none: no tokenization is applied and the input is passed directly to the BPE or SP model if set.
https://github.com/OpenNMT/Tokenizer/blob/master/docs/options.md

"""
import pyonmttok

tokenizer = pyonmttok.Tokenizer( 
    "conservative",
    bpe_model_path="",
    bpe_vocab_path="",  # Deprecated, use "vocabulary_path" instead.
    bpe_vocab_threshold=50,  # Deprecated, use "vocabulary_threshold" instead.
    vocabulary_path="",
    vocabulary_threshold=0,
    sp_model_path="",
    sp_nbest_size=0,
    sp_alpha=0.1,
    joiner="￭",
    joiner_annotate=False,
    joiner_new=False,
    spacer_annotate=False,
    spacer_new=False,
    case_feature=False,
    case_markup=False,
    no_substitution=False,
    preserve_placeholders=False,
    preserve_segmented_tokens=False,
    segment_case=False,
    segment_numbers=False,
    segment_alphabet_change=False,
    segment_alphabet=[])

text = "Hello World!"
tokens, features = tokenizer.tokenize(text)
print(tokens, features)
text = tokenizer.detokenize(tokens, features)
print(text)


