import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from configs import Configs
from model import DeepLoc
from dataset_loader import DatasetLoader
from utils import check_dir, load_file, cal_mrr, cal_map, cal_top
from torch.utils.data import DataLoader
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    model = DeepLoc(summary_emb=report_idx2summary_vec, desc_emb=report_idx2desc_vec,
                    code_idx2line_idx=code_idx2line_idx, line_idx2vec=line_idx2vec,
                    word_emb=word_idx2vec, n_kernels=N_KERNELS)
    model = model.to(device)
    print(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    train_set = DatasetLoader(train_dataset)
    train_data_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE,
                                   num_workers=1, pin_memory=True)

    best_MAP, best_epoch = 0, 0
    best_model_weight = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}')
        print('=' * 30)
        start_time = time.time()
        model, train_loss = _train(model, train_data_loader, optimizer, loss_func)
        train_time_elapse = time.time() - start_time

        print(f'== Train : {train_time_elapse // 60:.0f} min {train_time_elapse % 60:.0f} sec ==')
        print(f'Loss: {train_loss:.4f}\n')

        start_time = time.time()
        top_1, top_5, top_10, MAP, MRR = _test(model, val_dataset)
        time_elapse = time.time() - start_time

        print(f'==  Test : {time_elapse // 60:.0f} min {time_elapse % 60:.0f} sec  ==')
        print(f'MAP:   {MAP:.4f}')
        print(f'MRR:   {MRR:.4f}')
        print(f'TOP  1: {top_1:.4f}')
        print(f'TOP  5: {top_5:.4f}')
        print(f'TOP 10: {top_10:.4f}\n')

        if MAP >= best_MAP:
            best_MAP = MAP
            best_epoch = epoch
            best_model_weight = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch == EARLY_STOP:
            print('stop early.')
            break
    print(f'Best epoch: {best_epoch + 1}\n')
    model.load_state_dict(best_model_weight)
    return model


def _train(model, data_loader, optimizer, loss_func):
    model.train()
    train_loss = []
    for iteration, (report_idx, code_idx, recency_value, frequency_value, label) in enumerate(tqdm(data_loader, desc='train', ncols=80)):
        report_idx, code_idx, label = report_idx.to(device), code_idx.to(device), label.to(device)
        recency_value, frequency_value = recency_value.to(device), frequency_value.to(device)
        outputs = model(report_idx, code_idx, recency_value, frequency_value)
        loss = loss_func(outputs, label)
        train_loss.append(loss.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, sum(train_loss) / len(train_loss)


def _test(model, dataset):
    model.eval()
    results = []
    with torch.no_grad():
        for each_bug_data in tqdm(dataset, desc='test', ncols=80):
            each_bug_data = torch.FloatTensor(each_bug_data)
            report_idx, code_idx = each_bug_data[:, 0].long(), each_bug_data[:, 1].long()
            recency_value, frequency_value = each_bug_data[:, 2], each_bug_data[:, 3]
            label = each_bug_data[:, 4].long()

            report_idx, code_idx, label = report_idx.to(device), code_idx.to(device), label.to(device)
            recency_value, frequency_value = recency_value.to(device), frequency_value.to(device)

            output = model(report_idx, code_idx, recency_value, frequency_value)
            prob = torch.nn.functional.softmax(output, dim=1)
            results.append((label, prob[:, 1]))

    sorted_results = []
    for each_bug_result in results:
        labels = each_bug_result[0].tolist()
        probs = each_bug_result[1].tolist()
        sorted_labels_with_prob = list(sorted(zip(labels, probs), key=lambda x: x[1], reverse=True))
        sorted_labels = [str(item[0]) for item in sorted_labels_with_prob]
        sorted_results.append(sorted_labels)

    top_1, top_5, top_10 = cal_top(sorted_results)
    MAP, MRR = cal_map(sorted_results), cal_mrr(sorted_results)
    return top_1, top_5, top_10, MAP, MRR


def evaluate_model():
    model = DeepLoc(summary_emb=report_idx2summary_vec, desc_emb=report_idx2desc_vec,
                    code_idx2line_idx=code_idx2line_idx, line_idx2vec=line_idx2vec,
                    word_emb=word_idx2vec, n_kernels=N_KERNELS)
    model = model.to(device)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    top_1, top_5, top_10, MAP, MRR = _test(model, test_dataset)
    print()
    print('='*40 + 'Evaluation' + '='*40)
    print(f'MAP:   {MAP:.4f}')
    print(f'MRR:   {MRR:.4f}')
    print(f'TOP  1: {top_1:.4f}')
    print(f'TOP  5: {top_5:.4f}')
    print(f'TOP 10: {top_10:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, choices=['swt', 'tomcat', 'birt', 'jdt', 'eclipse'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_kernels', type=int, default=50)
    parser.add_argument('--num_neg', type=int, default=200)
    parser.add_argument('--early_stop', type=int, default=-1, help='Whether to stop training early.')
    parser.add_argument('--just_test', action='store_true', help='Only test with a pre-trained model.')
    args = parser.parse_args()

    PROJECT_NAME = args.project
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    N_KERNELS = args.n_kernels
    EARLY_STOP = args.early_stop
    NUM_NEG = args.num_neg
    JUST_TEST = args.just_test
    configs = Configs(PROJECT_NAME)

    print(args)

    dataset_dir = configs.dataset_dir
    vocab_dir = configs.vocabulary_dir

    train_dataset = load_file(f'{configs.dataset_dir}/train_dataset_neg_{NUM_NEG}.pkl')
    val_dataset = load_file(f'{configs.dataset_dir}/val_dataset_neg_{NUM_NEG}.pkl')
    test_dataset = load_file(f'{configs.dataset_dir}/test_dataset_neg_{NUM_NEG}.pkl')

    report_idx2summary_vec = load_file(configs.bugidx2summary_vec_path)
    report_idx2desc_vec = load_file(configs.bugidx2desc_vec_path)
    code_idx2line_idx = load_file(configs.code_idx2line_idx_path)
    line_idx2vec = load_file(configs.line_idx2vec_path)
    word_idx2vec = load_file(configs.word_idx2vec_path)

    model_save_dir = configs.model_save_dir

    if JUST_TEST:
        TRAINED_MODEL_PATH = f'{model_save_dir}/num_neg_{configs.num_neg}_{N_KERNELS}_{BATCH_SIZE}_{LR}.pth'
        if os.path.exists(TRAINED_MODEL_PATH):
            evaluate_model()
        else:
            raise ValueError(f'{TRAINED_MODEL_PATH} does not exist.')
    else:
        check_dir(model_save_dir)
        model_save_path = f'{model_save_dir}/num_neg_{configs.num_neg}_{N_KERNELS}_{BATCH_SIZE}_{LR}.pth'
        print(f'Model parameters saved in {model_save_path}')
        if os.path.exists(model_save_path):
            print(f'WARNING: {model_save_path} has existed!')

        deep_locator_model = train_model()
        torch.save(deep_locator_model.state_dict(), model_save_path)
        TRAINED_MODEL_PATH = model_save_path
        evaluate_model()
