from gensim.models.word2vec import Word2Vec
from configures import Configs
import _pickle as pickle
from Corpus.corpus import Corpus
import sys

project = sys.argv[1]
configs = Configs(project)
corpus_dir = configs.corpus_dir
corpus_dir = corpus_dir
corpus = Corpus(configs.corpus_dir, configs.report_path, configs.code_dir)

preprocessed_codes_dir = f'{corpus_dir}/preprocessed_codes'
preprocessed_reports_dir = f'{corpus_dir}/preprocessed_reports'
with open(f'{preprocessed_codes_dir}/preprocessed_code_tokens.pkl', 'rb') as f:
    code_corpus = pickle.load(f)

report_corpus = corpus.load_report_corpus(tag='total')
report_corpus = report_corpus['summary'] + ' ' + report_corpus['description']
report_corpus = report_corpus.tolist()
report_corpus = [s.split() for s in report_corpus]

sentences = code_corpus + report_corpus
model = Word2Vec(sentences=sentences, size=100, sg=1, workers=16, min_count=1)
model.wv.save_word2vec_format(f'{configs.data_dir}/{project}_word2vec.txt', binary=False)
