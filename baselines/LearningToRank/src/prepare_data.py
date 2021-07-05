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

    os.makedirs(f'../data/{project}/features')
    os.system(f'cp {DreamLoc_data_dir}/{project}/vocabulary/report_bugid2idx.pkl ../data/{project}/features/')
    os.system(f'cp {DreamLoc_data_dir}/{project}/vocabulary/bugidx2path_idx2valid_path_idx.pkl '
              f'../data/{project}/features/')
    os.system(f'cp {DreamLoc_data_dir}/{project}/vocabulary/commit_path2idx.pkl ../data/{project}/features/')
    os.system(f'cp {DreamLoc_data_dir}/{project}/features/tfidf_plus_sim.pkl ../data/{project}/features/')
    os.system(f'cp {DreamLoc_data_dir}/{project}/features/collaborative_filtering.pkl ../data/{project}/features/')
    os.system(f'cp {DreamLoc_data_dir}/{project}/features/fixing_frequency.pkl ../data/{project}/features/')
    os.system(f'cp {DreamLoc_data_dir}/{project}/features/fixing_recency.pkl ../data/{project}/features/')


