import argparse
import copy
import logging
import time
import torch
import torch.optim as optim
import sys
from configures import Configs
from collections import namedtuple
from evaluator import Evaluator
from torch.utils.data import DataLoader
from vocabulary import *
from dataset import load_dataset
from Model.model import DreamLoc
from Model.dataset_loader import DatasetLoader
from tqdm import tqdm
Formatted_pred = namedtuple('Formatted_pred', 'pred buggy_code_paths')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}\n')


def train(model, data_loader, optimizer):
    model.train()
    train_loss = []
    for iteration, (report_idx, pos_code_idx, neg_code_idx) in enumerate(tqdm(data_loader, desc='train', ncols=80)):
        report_idx, pos_code_idx, neg_code_idx = report_idx.to(device), pos_code_idx.to(device), neg_code_idx.to(device)
        loss = model(report_idx, pos_code_idx, neg_code_idx)
        loss = loss.mean()
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, sum(train_loss)/len(train_loss)


def cal_val_loss(model, data_loader):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for iteration, (report_idx, pos_code_idx, neg_code_idx) in enumerate(tqdm(data_loader, desc='val_loss', ncols=80)):
            report_idx, pos_code_idx, neg_code_idx = report_idx.to(device), pos_code_idx.to(device), neg_code_idx.to(device)
            loss = model(report_idx, pos_code_idx, neg_code_idx)
            val_loss.append(loss.mean().item())
    return sum(val_loss)/len(val_loss)


def test(model, dataset):
    model.eval()
    sub_list_len = BATCH_SIZE
    all_format_pred = []
    with torch.no_grad():
        for each_bug_data in tqdm(dataset, desc='test', ncols=80):
            r_c_pairs, code_paths, buggy_paths = each_bug_data
            r_c_pairs = torch.LongTensor(r_c_pairs).to(device)
            pred = []
            for i in range(0, r_c_pairs.shape[0], sub_list_len):
                r_idx = r_c_pairs[i: i+sub_list_len, 0]
                c_idx = r_c_pairs[i: i+sub_list_len, 1]
                sub_pred = model(r_idx, c_idx)
                pred += sub_pred.tolist()

            pred = list(zip(code_paths, pred))
            all_format_pred.append(
                Formatted_pred(pred=pred, buggy_code_paths=buggy_paths)
            )

    evaluator = Evaluator()
    ranked_predict = evaluator.rank(all_format_pred)
    hit_k, mean_ap, mean_rr = evaluator.evaluate(ranked_predict)
    return hit_k, mean_ap, mean_rr


def train_val_model(train_dataset, val_dataset, val_loss_dataset, model_save_path, matrix_bugidx2path_idx2valid_path_idx,
                    code_corpus_vec, code_word_idx2vec, snippet_idx2vec, code_snippet_idx2len, code_idx2len,
                    report_corpus_vec, report_word_idx2vec, report_word_idx2idf,
                    report_idx2cf, report_idx2ff, report_idx2fr, report_idx2sim, report_idx2tr, report_idx2cc):
    model = DreamLoc(FUSION_DENSE_DIM, RMM_DENSE_DIM, report_corpus_vec, report_word_idx2vec, report_word_idx2idf,
                     code_corpus_vec, code_word_idx2vec, snippet_idx2vec, code_snippet_idx2len,
                     code_idx2len, K_MAX_POOL, matrix_bugidx2path_idx2valid_path_idx,
                     IRFF_DENSE_DIM, report_idx2cf, report_idx2ff, report_idx2fr, report_idx2sim, report_idx2tr,
                     report_idx2cc, WITH_BIAS, WITH_DROPOUT, margin=1)

    model = torch.nn.DataParallel(model).to(device) if MULTI_GPU else model.to(device)
    print(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    train_set = DatasetLoader(train_dataset)
    train_data_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE,
                                   num_workers=1, pin_memory=True)

    best_map, best_epoch = 0, 0
    best_model_weight = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}')
        print('-' * 30)

        start_time = time.time()
        model, train_loss = train(model, train_data_loader, optimizer)
        train_time_elapse = time.time() - start_time

        print(f'== train : {train_time_elapse // 60:.0f} min {train_time_elapse % 60:.0f} sec ==')
        print(f'Loss: {train_loss:.4f}\n')

        start_time = time.time()
        val_hit_k, val_map, val_mrr = test(model, val_dataset)
        val_time_elapse = time.time() - start_time

        print(f'==  val : {val_time_elapse // 60:.0f} min {val_time_elapse % 60:.0f} sec  ==')
        print(f'MAP:   {val_map:.4f}')
        print(f'MRR:   {val_mrr:.4f}')
        for n, hit in enumerate(val_hit_k):
            print(f'hit_{n + 1}: {hit:.4f}')
        print()

        if val_map >= best_map:
            best_map = val_map
            best_epoch = epoch
            best_model_weight = copy.deepcopy(model.state_dict())

        elif EARLY_STOP and (epoch - best_epoch == 3):
            print('stop early.')
            break
    print(f'Best epoch: {best_epoch + 1}\n')

    if MULTI_GPU:
        model.load_state_dict(best_model_weight)
        best_model_weight = model.module.state_dict()
    torch.save(best_model_weight, model_save_path)


def evaluate(dataset, model_param, matrix_bugidx2path_idx2valid_path_idx,
             report_corpus_vec, report_word_idx2vec, report_word_idx2idf,
             code_corpus_vec, code_word_idx2vec, snippet_idx2vec, code_snippet_idx2len, code_idx2len,
             report_idx2cf, report_idx2ff, report_idx2fr, report_idx2sim, report_idx2tr, report_idx2cc):

    model = DreamLoc(FUSION_DENSE_DIM, RMM_DENSE_DIM, report_corpus_vec, report_word_idx2vec, report_word_idx2idf,
                     code_corpus_vec, code_word_idx2vec, snippet_idx2vec, code_snippet_idx2len,
                     code_idx2len, K_MAX_POOL, matrix_bugidx2path_idx2valid_path_idx,
                     IRFF_DENSE_DIM, report_idx2cf, report_idx2ff, report_idx2fr, report_idx2sim, report_idx2tr,
                     report_idx2cc, WITH_BIAS, WITH_DROPOUT, margin=1)
    model.load_state_dict(model_param)
    model = torch.nn.DataParallel(model).to(device) if MULTI_GPU else model.to(device)

    hit_k, mean_ap, mean_rr = test(model, dataset)
    print(f'MAP:   {mean_ap:.4f}')
    print(f'MRR:   {mean_rr:.4f}')
    for n, hit in enumerate(hit_k):
        print(f'hit_{n + 1}: {hit:.4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, choices=['swt', 'tomcat', 'birt', 'jdt', 'eclipse'])
    parser.add_argument('--rmm_dense_dim', type=int, default=100,
                        help='The dimension of Linear layer for snippet-aware report token encoding in Deep')
    parser.add_argument('--irff_dense_dim', type=int, default=20,
                        help='The dimension of Linear layer for IR features fusion in Wide.')
    parser.add_argument('--fusion_dense_dim', type=int, default=50,
                        help='The dimension of the Fusion in DreamLoc.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--k_max_pool', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--attention_type', type=str, default='dot',
                        choices=['dot', 'bilinear', 'dense', 'multi_head', 'dot_tfidf', 'cos_sim'])
    parser.add_argument('--calculate_val_loss', action='store_true',
                        help='Whether to split training dataset for calculating loss of validation.')
    parser.add_argument('--just_test', action='store_true', help='Only test with a pre-trained model.')
    parser.add_argument('--early_stop', action='store_true', help='Whether to stop training early.')
    parser.add_argument('--multiple_gpu', action='store_true', help='Whether to use multiple GPUs.')
    parser.add_argument('--with_bias', action='store_true', help='Whether to set bias as True in max_pool fusion.')
    parser.add_argument('--with_dropout', action='store_true')
    return parser.parse_args()


def main():
    configs = Configs(PROJECT_NAME)
    len_code_snippet = configs.len_code_snippet
    max_num_snippet = configs.max_num_snippet
    max_len_report = configs.max_len_report
    num_neg_sample = configs.num_neg_sample

    print(f'len_code_snippet = {len_code_snippet}')
    print(f'max_num_snippet = {max_num_snippet}')
    print(f'max_len_report = {max_len_report}')
    print(f'num_neg_sample = {num_neg_sample}\n')

    model_save_dir = configs.model_dir
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = f'{model_save_dir}/rmm_{RMM_DENSE_DIM}_irff_{IRFF_DENSE_DIM}_fusion_{FUSION_DENSE_DIM}_batchsize_{BATCH_SIZE}_' \
                      f'lr_{LR}_kpool_{K_MAX_POOL}.pth'

    if os.path.exists(model_save_path):
        logging.warning(f'{model_save_path} has existed !!!\n')
    print(f'model_param_path: {model_save_path}\n')

    dataset_dir = configs.dataset_dir
    vocab_dir = configs.vocab_dir

    code_corpus_vec, _ = load_code_corpus_vectors(vocab_dir)
    snippet_idx2vec = load_snippet_idx2vec(vocab_dir)
    code_word_idx2vec = load_word_idx2vec(vocab_dir, tag='code')
    code_snippet_idx2len = load_snippet_idx2len(vocab_dir)
    code_idx2len = load_code_idx2len(vocab_dir)

    report_corpus_vec, _ = load_report_corpus_vectors(vocab_dir)
    report_word_idx2vec = load_word_idx2vec(vocab_dir, tag='report')
    report_word_idx2idf = load_word_idx2idf(vocab_dir, tag='report')

    report_idx2cf = load_report_idx2cf(vocab_dir)
    report_idx2ff, report_idx2fr = load_report_idx2fixing_history(vocab_dir)
    report_idx2sim = load_report_idx2sim(vocab_dir)
    report_idx2tr = None
    report_idx2cc = load_report_idx2cc(vocab_dir)

    matrix_bugidx2path_idx2valid_path_idx = load_matrix_bugidx2path_idx2valid_path_idx(vocab_dir)

    train_dataset = load_dataset(dataset_dir, tag='train')
    val_dataset = load_dataset(dataset_dir, tag='val_metric')
    test_dataset = load_dataset(dataset_dir, tag='test')
    val_loss_dataset = None
    if VAL_LOSS:
        val_loss_dataset = load_dataset(dataset_dir, tag='val_loss')
    print(f'length of train dataset: {len(train_dataset)}')
    print(f'length of  val  dataset: {len(val_dataset)}')
    print(f'length of test  dataset: {len(test_dataset)}')

    if not JUST_TEST:
        train_val_model(train_dataset, val_dataset, val_loss_dataset, model_save_path, matrix_bugidx2path_idx2valid_path_idx,
                        code_corpus_vec, code_word_idx2vec, snippet_idx2vec, code_snippet_idx2len, code_idx2len,
                        report_corpus_vec, report_word_idx2vec, report_word_idx2idf,
                        report_idx2cf, report_idx2ff, report_idx2fr, report_idx2sim, report_idx2tr, report_idx2cc)

    logging.info(f'Evaluation Result:')
    model_param = torch.load(model_save_path, map_location=device)
    evaluate(test_dataset, model_param, matrix_bugidx2path_idx2valid_path_idx,
             report_corpus_vec, report_word_idx2vec, report_word_idx2idf,
             code_corpus_vec, code_word_idx2vec, snippet_idx2vec, code_snippet_idx2len, code_idx2len,
             report_idx2cf, report_idx2ff, report_idx2fr, report_idx2sim, report_idx2tr, report_idx2cc)
    print('-' * 100)
    print('\n\n')


if __name__ == '__main__':
    args = parse_args()
    LOG_FORMAT = "%(asctime)s - %(message)s"
    DATE_FORMAT = "%H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    data_dir = '../data'
    projects = ['eclipse.platform.swt', 'tomcat', 'birt', 'eclipse.platform.ui', 'eclipse.jdt.ui']

    PROJECT_NAME = args.project
    RMM_DENSE_DIM = args.rmm_dense_dim
    IRFF_DENSE_DIM = args.irff_dense_dim
    FUSION_DENSE_DIM = args.fusion_dense_dim
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    K_MAX_POOL = args.k_max_pool
    LR = args.lr
    ATTENTION_TYPE = args.attention_type
    VAL_LOSS = args.calculate_val_loss
    EARLY_STOP = args.early_stop
    MULTI_GPU = args.multiple_gpu
    JUST_TEST = args.just_test
    WITH_BIAS = args.with_bias
    WITH_DROPOUT = args.with_dropout

    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    print()
    main()
