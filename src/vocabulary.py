import itertools
import logging
import numpy as np
import os
import sys
sys.path.append('..')
from configures import Configs
from Corpus.corpus import Corpus
from gensim.models import KeyedVectors
from Features.collaborative_filtering import CollaborativeFiltering
from Features.fixing_history import FixingHistory
from Features.trace import Trace
from Features.cyclomatic_complexity import CyclomaticComplexity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from utils import save_file, load_file, sort_by_bugidx, list2matrix
LOG_FORMAT = "%(asctime)s - %(message)s"
DATE_FORMAT = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def generate_report_word2idf(report_corpus, save_dir):
    report_contents = report_corpus['summary'] + ' ' + report_corpus['description']
    vocab, idf = generate_word2idf(report_contents.tolist())
    save_file(f'{save_dir}/report_word2idx.pkl', vocab)
    save_file(f'{save_dir}/report_word2idf.pkl', idf)


def generate_code_word2idf(code_corpus, save_dir):
    code_contents = code_corpus['snippets'].tolist()
    vocab, idf = generate_word2idf(code_contents)
    save_file(f'{save_dir}/code_word2idx.pkl', vocab)
    save_file(f'{save_dir}/code_word2idf.pkl', idf)


def generate_word2idf(data):
    tfidf = TfidfVectorizer().fit(data)
    vocab = tfidf.vocabulary_
    idf = tfidf.idf_

    for w in vocab:
        vocab[w] = vocab[w] + 1
    vocab['<pad>'] = 0
    idf = np.insert(idf, 0, 0.0)

    return vocab, idf


def load_word2idx(tag, save_dir):
    assert tag in ('report', 'code')
    vocab = load_file(f'{save_dir}/{tag}_word2idx.pkl')
    return vocab


def load_word_idx2idf(vocab_dir, tag):
    assert tag in ('report', 'code')
    vocab = load_file(f'{vocab_dir}/{tag}_word2idf.pkl')
    if len(vocab.shape) != 2:
        vocab = np.expand_dims(vocab, axis=1)
    return vocab


def generate_report_word_idx2vec(enwiki_model, camel_word2split, save_dir):
    word2idx = load_word2idx('report', save_dir)
    word_idx2vec = _generate_word_idx2vec(camel_word2split, word2idx, enwiki_model)
    save_file(f'{save_dir}/report_word2vec.pkl', word_idx2vec)


def generate_code_word_idx2vec(enwiki_model, camel_word2split, save_dir):
    word2idx = load_word2idx('code', save_dir)
    word_idx2vec = _generate_word_idx2vec(camel_word2split, word2idx, enwiki_model)
    save_file(f'{save_dir}/code_word2vec.pkl', word_idx2vec)


def load_word_idx2vec(vocab_dir, tag):
    assert tag in ('code', 'report')
    word_idx2vec = load_file(f'{vocab_dir}/{tag}_word2vec.pkl')
    return word_idx2vec


def _generate_word_idx2vec(camel_word2split, word2idx, enwiki_model):
    emb_dim = enwiki_model.vector_size
    words = set(word2idx.keys())
    camel_words = set(camel_word2split.keys())
    non_camel_words = words - camel_words

    word_emb_vocab = {}
    oov_record = {}
    # how many words in project's vocab do not exist in wiki vocab
    oov_non_camel_words = []
    # how many camel words from test reports and codes whose split tokens do not exist in project's vocab.
    # split tokens of camel words from train and val corpus are all in project's vocab
    oov_camel_words = []

    for w in non_camel_words:
        if w in enwiki_model.wv:
            word_emb_vocab[w] = enwiki_model[w]
        else:
            word_emb_vocab[w] = np.random.uniform(-0.25, 0.25, emb_dim)
            oov_non_camel_words.append(w)

    for cw in camel_word2split:
        split_tokens = camel_word2split[cw]
        vectors = []
        for token in split_tokens:
            if token in word_emb_vocab:
                vectors.append(word_emb_vocab[token])
            # else:
            #     vectors.append(np.random.uniform(-0.25, 0.25, emb_dim))
            #     oov_camel_words.append(cw)
        if len(vectors) == 0:
            vectors.append(np.random.uniform(-0.25, 0.25, emb_dim))
            oov_camel_words.append(cw)
        camel_vec = np.mean(vectors, axis=0)
        word_emb_vocab[cw] = camel_vec
    oov_camel_words = list(set(oov_camel_words))

    word_emb_vocab['<pad>'] = np.zeros(emb_dim)
    word_idx2vec = np.zeros((len(word_emb_vocab), emb_dim))
    for w in word2idx:
        word_idx2vec[word2idx[w]] = word_emb_vocab[w]

    oov_record['non_camel_words_in_train_val_corpus'] = oov_non_camel_words
    oov_record['camel_words_in_test_corpus'] = oov_camel_words
    logging.info(f'number of non_camel_words: {len(non_camel_words)}')
    logging.info(f'number of camel_words: {len(camel_words)}')
    logging.info(f'number of non_camel_words OOV: {len(oov_non_camel_words)}')
    logging.info(f'number of camel_words OOV: {len(oov_camel_words)}')
    return word_idx2vec


def vectorize_code_corpus(code_corpus, word2idx, save_dir, max_num_snippet=50):
    """
    code_corpus_vectors: np.array, shape: (num_snippets, max_num_snippet)
    [
        [snippet_idx, ...],
        ...
        [snippet_idx, ...]
    ]

    snippet_idx2vec: np.array, shape: (num_snippets, max_len_snippet)
    [
        [word_idx...]
        ...
        [word_idx...]
    ]
    """
    code_corpus_vectors = []
    snippet_len = [0]
    snippet_num = []
    snippet_vectors = [[0]]  # [0] is used to pad
    snippet_idx = 1
    codes = code_corpus['snippets'].tolist()
    for code in tqdm(codes, desc='vectorizing code_corpus', ncols=100):
        code_vec = []
        snippets = code.split('\n')
        for snippet_count, snippet in enumerate(snippets):
            if snippet_count == max_num_snippet:
                break
            snippet_vev = []
            snippet_words = snippet.split()
            for i, word in enumerate(snippet_words):
                snippet_vev.append(word2idx[word])
            snippet_vectors.append(snippet_vev)
            snippet_len.append(len(snippet_vev))
            code_vec.append(snippet_idx)
            snippet_idx += 1
        code_corpus_vectors.append(code_vec)
        snippet_num.append(len(code_vec))
    assert len(code_corpus) == len(code_corpus_vectors)

    pad_code_corpus_vectors = list(itertools.zip_longest(*code_corpus_vectors, fillvalue=0))
    code_corpus_matrix = np.array(pad_code_corpus_vectors).transpose((1, 0))

    pad_snippet_vectors = list(itertools.zip_longest(*snippet_vectors, fillvalue=0))
    snippet_vector_matrix = np.array(pad_snippet_vectors).transpose((1, 0))

    snippet_len = np.array(snippet_len)
    snippet_num = np.array(snippet_num)
    save_file(f'{save_dir}/code_corpus_vectors.pkl', code_corpus_matrix)
    save_file(f'{save_dir}/snippet_idx2vec.pkl', snippet_vector_matrix)
    save_file(f'{save_dir}/snippet_idx2len.pkl', snippet_len)
    save_file(f'{save_dir}/code_idx2len.pkl', snippet_num)

    commit_paths = code_corpus['commit'] + '/' + code_corpus['path']
    commit_paths = commit_paths.tolist()
    commit_path2idx = dict(list(zip(commit_paths, range(len(commit_paths)))))
    save_file(f'{save_dir}/commit_path2idx.pkl', commit_path2idx)


def generate_code_idx2valid_code_idx(commit2path2commit_path, commit_path2idx, commit2bugid, bugid2idx, vocab_dir):
    bugidx2path2idx = {}
    bugidx2path_idx2valid_path_idx = {}
    for commit in commit2path2commit_path:
        if commit not in commit2bugid:
            continue  # some reports' commit don't have new file compared with last commit.
        bugid = commit2bugid[commit]
        bugidx = bugid2idx[bugid]
        path2commit_path = commit2path2commit_path[commit]
        path2idx = {}
        path_idx2commit_path_idx = {}
        for idx, (path, commit_path) in enumerate(path2commit_path.items()):  # bug_id -> bug_idx -> all java idx in current commit -> idx in code corpus -> all java content
            path2idx[path] = idx
            path_idx2commit_path_idx[idx] = commit_path2idx[commit_path]  # idx in code corpus
        bugidx2path2idx[bugidx] = path2idx
        bugidx2path_idx2valid_path_idx[bugidx] = path_idx2commit_path_idx

    # a matrix for bugidx2path_idx2valid_path_idx
    matrix_bugidx2path_idx2valid_path_idx = []
    for idx in range(len(bugidx2path_idx2valid_path_idx)):
        path_idx2valid_path_idx = bugidx2path_idx2valid_path_idx[idx]
        matrix_bugidx2path_idx2valid_path_idx.append(list(path_idx2valid_path_idx.values()))

    matrix_bugidx2path_idx2valid_path_idx = list(itertools.zip_longest(*matrix_bugidx2path_idx2valid_path_idx,
                                                                       fillvalue=0))
    matrix_bugidx2path_idx2valid_path_idx = np.array(matrix_bugidx2path_idx2valid_path_idx).transpose((1, 0))

    save_file(f'{vocab_dir}/bugidx2path2idx.pkl', bugidx2path2idx)
    save_file(f'{vocab_dir}/bugidx2path_idx2valid_path_idx.pkl', bugidx2path_idx2valid_path_idx)
    save_file(f'{vocab_dir}/matrix_bugidx2path_idx2valid_path_idx.pkl', matrix_bugidx2path_idx2valid_path_idx)


def load_bugidx2path2idx(vocab_dir):
    bugidx2path2idx = load_file(f'{vocab_dir}/bugidx2path2idx.pkl')
    return bugidx2path2idx


def load_bugidx2path_idx2valid_path_idx(vocab_dir):
    bugidx2path_idx2valid_path_idx = load_file(f'{vocab_dir}/bugidx2path_idx2valid_path_idx.pkl')
    return bugidx2path_idx2valid_path_idx


def load_matrix_bugidx2path_idx2valid_path_idx(vocab_dir):
    matrix_bugidx2path_idx2valid_path_idx = load_file(f'{vocab_dir}/matrix_bugidx2path_idx2valid_path_idx.pkl')
    return matrix_bugidx2path_idx2valid_path_idx


def generate_report_idx2sim(bugidx2path_idx2valid_path_idx, feature_dir, vocab_dir):
    bugidx2sim = []
    tfidf_plus_sim = load_file(f'{feature_dir}/tfidf_plus_sim.pkl')
    for bugidx, path_idx2valid_path_idx in bugidx2path_idx2valid_path_idx.items():  # valid_path_idx is the idx in code corpus
        path_idx2sim = []
        for path_idx, valid_path_idx in path_idx2valid_path_idx.items():
            path_idx2sim.append((path_idx, tfidf_plus_sim[bugidx][valid_path_idx]))
        path_idx2sim = list(sorted(path_idx2sim, key=lambda x: x[0]))
        sim = [item[1] for item in path_idx2sim]
        bugidx2sim.append((bugidx, sim))
    bugidx2sim = list(sorted(bugidx2sim, key=lambda x: x[0]))
    sim = [item[1] for item in bugidx2sim]
    pad_sim = list(itertools.zip_longest(*sim, fillvalue=0))
    sim = np.array(pad_sim).transpose((1, 0))
    save_file(f'{vocab_dir}/report_idx2tfidf_plus_sim.pkl', sim)


def load_report_idx2sim(vocab_dir):
    sim = load_file(f'{vocab_dir}/report_idx2tfidf_plus_sim.pkl')
    return sim


def generate_report_idx2cf(collective_filtering_score, bugid2idx, bugidx2path2idx, vocab_dir):
    """np.array shape: (num_reports, num_code_paths). with pad for each bugid"""
    cf_matrix = []
    for bugid, path2score in collective_filtering_score.items():
        bugidx = bugid2idx[int(bugid)]
        path2idx = bugidx2path2idx[bugidx]
        cf = [0.0] * len(path2idx)
        for path, score in path2score.items():
            path_idx = path2idx[path]
            cf[path_idx] = score
        cf_matrix.append((bugidx, cf))
    cf_matrix = list(sorted(cf_matrix, key=lambda x: x[0]))
    cf_matrix = [item[1] for item in cf_matrix]
    cf_matrix = list(itertools.zip_longest(*cf_matrix, fillvalue=0.0))
    cf_matrix = np.array(cf_matrix).transpose((1, 0))
    save_file(f'{vocab_dir}/report_idx2collective_filtering.pkl', cf_matrix)


def load_report_idx2cf(vocab_dir):
    cf = load_file(f'{vocab_dir}/report_idx2collective_filtering.pkl')
    return cf


def generate_report_idx2fixing_history(fixing_frequency, fixing_recency, bugid2idx, bugidx2path2idx, vocab_dir):
    """np.array shape: (num_reports, num_code_paths)"""
    ff_matrix = []
    fr_matrix = []
    for bugid in fixing_frequency:
        bugidx = bugid2idx[int(bugid)]
        path2idx = bugidx2path2idx[bugidx]
        ff = [0.0] * len(path2idx)
        fr = [0.0] * len(path2idx)
        for path, score in fixing_frequency[bugid].items():
            if path not in path2idx:
                continue
            ff[path2idx[path]] = score
        for path, score in fixing_recency[bugid].items():
            if path not in path2idx:
                continue
            fr[path2idx[path]] = score
        ff_matrix.append((bugidx, ff))
        fr_matrix.append((bugidx, fr))
    ff_matrix = sort_by_bugidx(ff_matrix)
    ff_matrix = list2matrix(ff_matrix)
    fr_matrix = sort_by_bugidx(fr_matrix)
    fr_matrix = list2matrix(fr_matrix)
    save_file(f'{vocab_dir}/report_idx2fixing_frequency.pkl', ff_matrix)
    save_file(f'{vocab_dir}/report_idx2fixing_recency.pkl', fr_matrix)


def load_report_idx2fixing_history(vocab_dir):
    ff = load_file(f'{vocab_dir}/report_idx2fixing_frequency.pkl')
    fr = load_file(f'{vocab_dir}/report_idx2fixing_recency.pkl')
    return ff, fr


def generate_report_idx2trace(trace, bugid2idx, bugidx2path2idx, vocab_dir):
    tr_matrix = []
    for bugid in trace:
        bugidx = bugid2idx[bugid]
        path2idx = bugidx2path2idx[bugidx]
        tr = [0.2] * len(path2idx)
        for path in trace[bugid]:
            tr[path2idx[path]] = 1.0
        tr_matrix.append((bugidx, tr))
    tr_matrix = sort_by_bugidx(tr_matrix)
    tr_matrix = list2matrix(tr_matrix)
    save_file(f'{vocab_dir}/report_idx2trace.pkl', tr_matrix)


def load_report_idx2trace(vocab_dir):
    tr = load_file(f'{vocab_dir}/report_idx2trace.pkl')
    return tr


def generate_report_idx2cc(cyclomatic_complexity, commit_path2idx, bugidx2path2idx,
                           bugidx2path_idx2valid_path_idx, vocab_dir):
    cc_matrix = []
    valid_path_idx2commit_path = dict(zip(commit_path2idx.values(), commit_path2idx.keys()))
    for bugidx, path2idx in bugidx2path2idx.items():
        path_idx2valid_path_idx = bugidx2path_idx2valid_path_idx[bugidx]
        cc = [0.0] * len(path2idx)
        for path, idx in path2idx.items():
            valid_path_idx = path_idx2valid_path_idx[idx]
            valid_path = valid_path_idx2commit_path[valid_path_idx]
            cc[idx] = cyclomatic_complexity[valid_path]
        cc_matrix.append((bugidx, cc))
    cc_matrix = sort_by_bugidx(cc_matrix)
    cc_matrix = list2matrix(cc_matrix)
    save_file(f'{vocab_dir}/report_idx2cyclomatic_complexity.pkl', cc_matrix)


def load_report_idx2cc(vocab_dir):
    cc = load_file(f'{vocab_dir}/report_idx2cyclomatic_complexity.pkl')
    return cc


def load_code_corpus_vectors(vocab_dir):
    code_corpus_vectors = load_file(f'{vocab_dir}/code_corpus_vectors.pkl')
    commit_path2idx = load_commit_path2idx(vocab_dir)
    return code_corpus_vectors, commit_path2idx


def load_commit_path2idx(vocab_dir):
    commit_path2idx = load_file(f'{vocab_dir}/commit_path2idx.pkl')
    return commit_path2idx


def load_snippet_idx2vec(vocab_dir):
    snippet_idx2vector = load_file(f'{vocab_dir}/snippet_idx2vec.pkl')
    return snippet_idx2vector


def load_code_idx2len(vocab_dir):
    code_idx2len = load_file(f'{vocab_dir}/code_idx2len.pkl')
    if len(code_idx2len.shape) != 2:
        code_idx2len = np.expand_dims(code_idx2len, axis=1)
    return code_idx2len


def load_snippet_idx2len(vocab_dir):
    snippet_len = load_file(f'{vocab_dir}/snippet_idx2len.pkl')
    if len(snippet_len.shape) != 2:
        snippet_len = np.expand_dims(snippet_len, axis=1)
    return snippet_len


def vectorize_report_corpus(report_corpus, word2idx, save_dir, max_len):
    """
    if the length of report large than max_len, truncate and append keywords.

    report_{tag}_corpus_vectors: np.array  shape: (num_report, max_len)
    [
        [word_idx_1, ..., word_idx_MaxLen],
        ...,
        [word_idx_1, ..., word_idx_MaxLen]
    ]

    report_{tag}_bugid2idx: dict
    {
        bug_id: report_idx_in_corpus_vectors
    }

    """
    bug_id2idx = {}
    report_corpus_vector = []

    for idx, report in enumerate(tqdm(report_corpus.itertuples(), desc=f'vectorizing report_corpus', ncols=100)):
        report_vec = []
        content_words = f'{report.summary} {report.description}'.split()
        if len(content_words) > max_len:  # truncate and append keywords
            keywords = list(set(f'{report.keywords_summary} {report.keywords_description}'.split()))
            if len(keywords) > max_len:
                keywords = list(sorted(keywords, key=lambda k: len(k), reverse=True))[:max_len]
            content_words = content_words[:max_len-len(keywords)] + keywords
            assert len(content_words) == max_len
        for i, word in enumerate(content_words):
            report_vec.append(word2idx[word])
        bug_id2idx[report.bug_id] = idx
        report_corpus_vector.append(report_vec)

    pad_report_vectors = list(itertools.zip_longest(*report_corpus_vector, fillvalue=0))
    report_vector_matrix = np.array(pad_report_vectors).transpose((1, 0))

    save_file(f'{save_dir}/report_corpus_vectors.pkl', report_vector_matrix)
    save_file(f'{save_dir}/report_bugid2idx.pkl', bug_id2idx)


def load_report_corpus_vectors(save_dir):
    report_vector_matrix = load_file(f'{save_dir}/report_corpus_vectors.pkl')
    bugid2idx = load_file(f'{save_dir}/report_bugid2idx.pkl')
    return report_vector_matrix, bugid2idx


def main(project_name):
    configs = Configs(project_name)
    logging.info(f"feature_dir: {configs.feature_dir}\n")

    corpus_dir = configs.corpus_dir
    feature_dir = configs.feature_dir
    vocab_dir = configs.vocab_dir
    if not os.path.exists(vocab_dir):
        os.mkdir(vocab_dir)
    logging.info(f'vocabulary dir: {vocab_dir}')
    corpus = Corpus(corpus_dir)
    code_corpus = corpus.load_code_corpus()
    report_corpus = corpus.load_report_corpus('total')

    # 1.word2idx and word2idf
    logging.info(f'generating word2idx and word2idf...')
    logging.info(f'NOTE: add PAD in word2idx and word2idf at index 0.')
    generate_report_word2idf(report_corpus, vocab_dir)
    generate_code_word2idf(code_corpus, vocab_dir)

    # 2.word2vec
    # enwiki_path = f'../data/enwiki_20180420_100d.txt'
    # enwiki_model = KeyedVectors.load_word2vec_format(enwiki_path, binary=False)
    print(f'Using project word2vec.')
    word2vec_path = f'../data/{project_name}_word2vec.txt'
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    report_camel_word2split = corpus.load_report_camel_word_record('total')
    code_camel_word2split = corpus.load_code_camel_word_record()
    logging.info(f'generating report word2vec...')
    generate_report_word_idx2vec(word2vec_model, report_camel_word2split, vocab_dir)
    logging.info(f'generating code word2vec...')
    generate_code_word_idx2vec(word2vec_model, code_camel_word2split, vocab_dir)

    # 3.vectorize code and report corpus
    code_word2idx = load_word2idx(tag='code', save_dir=vocab_dir)
    vectorize_code_corpus(code_corpus, code_word2idx, vocab_dir, max_num_snippet=configs.max_num_snippet)

    report_word2idx = load_word2idx(tag='report', save_dir=vocab_dir)
    vectorize_report_corpus(report_corpus, report_word2idx, vocab_dir, max_len=configs.max_len_report)

    # 4. code_idx2valid_code_idx
    logging.info('generating code_idx to valid_code_idx...')
    _, bugid2idx = load_report_corpus_vectors(vocab_dir)
    commit2path2commit_path = corpus.load_commit2commit_code_paths()
    commit_path2idx = load_commit_path2idx(vocab_dir)

    commit2bugid = dict(zip(report_corpus['commit'].tolist(), report_corpus['bug_id'].tolist()))
    generate_code_idx2valid_code_idx(commit2path2commit_path, commit_path2idx, commit2bugid, bugid2idx, vocab_dir)
    bugidx2path2idx = load_bugidx2path2idx(vocab_dir)
    bugidx2path_idx2valid_path_idx = load_bugidx2path_idx2valid_path_idx(vocab_dir)

    # 5.feature vocabularies
    logging.info('generating feature vocabularies...')
    generate_report_idx2sim(bugidx2path_idx2valid_path_idx, feature_dir, vocab_dir)

    cf = CollaborativeFiltering(feature_dir)
    collective_filtering_score = cf.load()
    generate_report_idx2cf(collective_filtering_score, bugid2idx, bugidx2path2idx, vocab_dir)

    fixing_history = FixingHistory(feature_dir)
    fixing_frequency = fixing_history.load_fixing_frequency()
    fixing_recency = fixing_history.load_fixing_recency()
    generate_report_idx2fixing_history(fixing_frequency, fixing_recency, bugid2idx, bugidx2path2idx, vocab_dir)

    # trace = Trace(feature_dir)
    # tr = trace.load()
    # generate_report_idx2trace(tr, bugid2idx, bugidx2path2idx, vocab_dir)

    cyclomatic_complexity = CyclomaticComplexity(feature_dir)
    cc = cyclomatic_complexity.load()
    generate_report_idx2cc(cc, commit_path2idx, bugidx2path2idx, bugidx2path_idx2valid_path_idx, vocab_dir)


if __name__ == '__main__':
    project_name = sys.argv[1]
    main(project_name)
