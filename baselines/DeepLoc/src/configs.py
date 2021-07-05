class Configs:
    def __init__(self, project_name):
        project_id = {
            'swt': 0, 'tomcat': 1, 'birt': 2, 'eclipse': 3, 'jdt': 4
        }
        assert project_name in project_id
        data_dir = '../data'
        self.data_dir = data_dir
        self.report_path = f'{data_dir}/{project_name}/{project_name}.xml'
        self.collected_code_dir = f'{data_dir}/{project_name}/corpus/collected_codes'
        self.report_corpus_dir = f'{data_dir}/{project_name}/corpus/preprocessed_reports'
        self.code_corpus_path = f'{data_dir}/{project_name}/corpus/code_corpus.csv'
        self.code_main_node_lines_path = f'{data_dir}/{project_name}/corpus/src_main_node_lines.pkl'
        self.commit2code_commit_paths_path = f'{data_dir}/{project_name}/corpus/commit2commit_code_paths.pkl'
        self.dataset_dir = f'{data_dir}/{project_name}/dataset'
        self.model_save_dir = f'{data_dir}/{project_name}/model'
        self.vocabulary_dir = f'{data_dir}/{project_name}/vocabulary'

        self.sent2vec_dir = f'/home/LAB/qibh/Downloads/sent2vec-master'
        self.sent2vec_model_corpus_path = f'{self.report_corpus_dir}/description_sentences.txt'
        self.sent2vec_model_path = f'{self.vocabulary_dir}/sent2vec_model'
        self.bugid2idx_path = f'{data_dir}/{project_name}/vocabulary/bugid2idx.pkl'
        self.bugidx2desc_vec_path = f'{self.vocabulary_dir}/bugidx2desc_vec.pkl'
        self.bugidx2summary_vec_path = f'{self.vocabulary_dir}/bugidx2summary_vec.pkl'

        self.word2vec_path = f'{data_dir}/{project_name}/vocabulary/{project_name}_word2vec.txt'
        self.word2idx_path = f'{data_dir}/{project_name}/vocabulary/word2idx.pkl'
        self.word_idx2vec_path = f'{data_dir}/{project_name}/vocabulary/word_idx2vec.pkl'
        self.commit_path2code_idx_path = f'{data_dir}/{project_name}/vocabulary/commit_path2code_idx.pkl'
        self.code_commit_path2code_idx_path = f'{self.vocabulary_dir}/code_commit_path2code_idx.pkl'
        self.code_idx2line_idx_path = f'{self.vocabulary_dir}/code_idx2line_idx.pkl'
        self.line_idx2vec_path = f'{self.vocabulary_dir}/line_idx2vec.pkl'

        self.fixing_frequency_path = f'{data_dir}/{project_name}/features/fixing_frequency.pkl'
        self.fixing_recency_path = f'{data_dir}/{project_name}/features/fixing_recency.pkl'
        self.tfidf_sim_path = f'{data_dir}/{project_name}/features/sur_sim.pkl'

        self.num_neg = 200
