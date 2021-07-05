import sys
import os

if __name__ == '__main__':
    project = sys.argv[1]
    DreamLoc_data_dir = f'../../../DreamLoc/data'

    os.makedirs(f'../data/{project}/corpus')
    os.system(f'cp -r {DreamLoc_data_dir}/{project}/corpus/collected_codes ../data/{project}/corpus/')
    os.system(f'cp -r {DreamLoc_data_dir}/{project}/corpus/preprocessed_reports ../data/{project}/corpus/')
    os.system(f'cp {DreamLoc_data_dir}/{project}/corpus/preprocessed_codes/commit2commit_code_paths.pkl '
              f'../data/{project}/corpus/')

    os.makedirs(f'../data/{project}/vocabulary')
    os.system(f'cp {DreamLoc_data_dir}/{project}_word2vec.txt ../data/{project}/vocabulary/')

    os.makedirs(f'../data/{project}/features')
    os.system(f'cp {DreamLoc_data_dir}/{project}/features/fixing_frequency.pkl ../data/{project}/features/')
    os.system(f'cp {DreamLoc_data_dir}/{project}/features/fixing_recency.pkl ../data/{project}/features/')


