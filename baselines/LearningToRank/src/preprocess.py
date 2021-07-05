import re
import inflection
from string import punctuation
from assets import *
filter_words_set = stop_words.union(java_keywords)


def preprocess(sentence):
    tokens = tokenize(sentence)
    split_tokens, camel_word_split_record = split_camelcase(tokens)
    normalized_tokens = normalize(split_tokens)
    filter_tokens = filter_words(normalized_tokens)
    return filter_tokens


def tokenize(sent):
    filter_sent = re.sub('[^a-zA-Z]', ' ', sent)
    tokens = filter_sent.split()
    return tokens


def split_camelcase(tokens, retain_camelcase=True):
    """
    :param tokens: [str]
    :param retain_camelcase: if True, the corpus will retain camel words after splitting them.
    :return:
    """
    def split_by_punc(token):
        new_tokens = []
        split_toks = re.split(fr'[{punctuation}]+', token)
        if len(split_toks) > 1:
            return_tokens.remove(token)
            for st in split_toks:
                if not st:  # st may be '', e.g. tok = '*' then split_toks = ['', '']
                    continue
                return_tokens.append(st)
                new_tokens.append(st)
        return new_tokens

    def split_by_camel(token):
        camel_split = inflection.underscore(token).split('_')
        if len(camel_split) > 1:
            if any([len(cs) > 2 for cs in camel_split]):
                return_tokens.extend(camel_split)
                camel_word_split_record[token] = camel_split
                if not retain_camelcase:
                    return_tokens.remove(token)

    camel_word_split_record = {}  # record camel words and their generation e.g. CheckBuff: [check, buff]
    # return_tokens = tokens[:]
    return_tokens = []
    for tok in tokens:
        return_tokens.append(tok)
        if not bool(re.search(r'[a-zA-Z]', tok)):
            continue
        new_tokens = split_by_punc(tok)
        new_tokens = new_tokens if new_tokens else [tok]
        for nt in new_tokens:
            split_by_camel(nt)
    return return_tokens, camel_word_split_record


def normalize(tokens):
    normalized_tokens = [tok.lower() for tok in tokens]
    return normalized_tokens


def filter_words(tokens):
    tokens = [tok for tok in tokens if tok not in filter_words_set and len(tok) > 1]
    return tokens
