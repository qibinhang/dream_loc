import inflection
import re
import sys
sys.path.append('..')
from Corpus.assets import *
from string import punctuation, digits


class Preprocessor:
    """
    NOTE:
        retain camelcase after splitting camelcase.
        no stem
    """

    def __init__(self):
        super(Preprocessor, self).__init__()
        self.filter_words_set = stop_words.union(java_keywords)
        self.punct_num_table = str.maketrans({c: None for c in punctuation + digits})

    def preprocess_report(self, sentence, tag):
        assert tag in ('summary', 'description')
        tokens, camel_word_split_record = self._preprocess(sentence)
        tokens = tokens[1:] if tag == 'summary' else tokens  # if sentence is summary, remove first token 'bug'
        return tokens, camel_word_split_record

    def preprocess_code(self, code):
        code = self.remove_import_and_package(code)
        tokens, camel_word_split_record = self._preprocess(code)
        return tokens, camel_word_split_record

    def preprocess_code_with_multiprocess(self, items):
        """idx for multiprocessing"""
        code, idx = items
        code = self.remove_import_and_package(code)
        tokens, camel_word_split_record = self._preprocess(code)
        return tokens, camel_word_split_record, idx

    @staticmethod
    def remove_import_and_package(code):
        code_lines = code.split('\n')
        package_line_num = 0
        last_import_line_num = 0
        for i, line in enumerate(code_lines):
            if line.startswith('import '):
                last_import_line_num = i
            elif line.startswith('package '):
                package_line_num = i
            elif last_import_line_num != 0 and line:
                break
        new_start_line_num = max(last_import_line_num + 1, package_line_num + 1)
        code = '\n'.join(code_lines[new_start_line_num:])
        return code

    def _preprocess(self, sentence, record_camel_word_split=True):
        tokens = self.tokenize(sentence)
        split_tokens, camel_word_split_record = self.split_camelcase(tokens)
        normalized_tokens = self.normalize(split_tokens)
        filter_tokens = self.filter_words(normalized_tokens)
        processed_camel_word_split_record = {}
        if record_camel_word_split:
            processed_camel_word_split_record = self._record_camel_word_split(camel_word_split_record)

        return filter_tokens, processed_camel_word_split_record

    def _record_camel_word_split(self, camel_word_split_record):
        processed_camel_word_split_record = {}
        for camel_word, split_camel in camel_word_split_record.items():
            camel_word = self.normalize([camel_word])
            camel_word = self.filter_words(camel_word)

            split_camel = self.normalize(split_camel)
            split_camel = self.filter_words(split_camel)
            if camel_word and split_camel:
                processed_camel_word_split_record[camel_word[0]] = split_camel
        return processed_camel_word_split_record

    @staticmethod
    def tokenize(sent):
        filter_sent = re.sub('[^a-zA-Z]', ' ', sent)
        tokens = filter_sent.split()
        return tokens

    @staticmethod
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

    @staticmethod
    def normalize(tokens):
        normalized_tokens = [tok.lower() for tok in tokens]
        return normalized_tokens

    def filter_words(self, tokens):
        tokens = [tok for tok in tokens if tok not in self.filter_words_set and len(tok) > 1]
        return tokens

    def process_code_name(self, code_name, recombine=False, record_camel_word_split=False):
        """split code_name and recombine"""
        tokens, camel_word_split_record = self.split_camelcase([code_name])
        if recombine and len(tokens) > 3:  # like [AaaBbbCcc, aaa, bbb, ccc]
            recombine_tokens = [f'{tokens[i]}{tokens[i+1]}' for i in range(1, len(tokens)-1)]
            tokens += recombine_tokens

        tokens = self.normalize(tokens)
        tokens = self.filter_words(tokens)
        if record_camel_word_split:
            processed_camel_word_split_record = {}
            for camel_word, split_camel in camel_word_split_record.items():
                camel_word = self.normalize([camel_word])

                split_camel = self.normalize(split_camel)
                split_camel = self.filter_words(split_camel)
                processed_camel_word_split_record[camel_word[0]] = split_camel
            return tokens, processed_camel_word_split_record
        else:
            return tokens
