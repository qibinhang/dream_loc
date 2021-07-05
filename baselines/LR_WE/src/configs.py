class Configs:
    def __init__(self, project_name):
        project_id = {
            'swt': 0, 'tomcat': 1, 'birt': 2, 'eclipse': 3, 'jdt': 4
        }
        assert project_name in project_id
        data_dir = '../data'
        self.data_dir = data_dir
        self.collected_code_dir = f'{data_dir}/{project_name}/corpus/collected_codes'
        self.src_api_path = f'{data_dir}/{project_name}/features/src_api.pkl'
        self.api_desc_path = f'{data_dir}/api_desc/{project_name}_api.csv'
        self.report_corpus_dir = f'{data_dir}/{project_name}/corpus/preprocessed_reports'
        self.code_corpus_dir = f'{data_dir}/{project_name}/corpus/preprocessed_codes'
        self.commit2code_commit_paths_path = f'{data_dir}/{project_name}/corpus/commit2commit_code_paths.pkl'
        self.word2vec_path = f'{data_dir}/{project_name}/{project_name}_word2vec.txt'
        self.word_vocab_path = f'{data_dir}/{project_name}/features/sur_sim_vocab.pkl'

        self.api_enrich_sim_path = f'{data_dir}/{project_name}/features/api_enrich_sim.pkl'
        self.class_name_sim_path = f'{data_dir}/{project_name}/features/class_name_sim.pkl'
        self.surface_sim_path = f'{data_dir}/{project_name}/features/tfidf_plus_sim.pkl'
        self.fixing_frequency_path = f'{data_dir}/{project_name}/features/fixing_frequency.pkl'
        self.fixing_recency_path = f'{data_dir}/{project_name}/features/fixing_recency.pkl'
        self.collaborative_filtering_path = f'{data_dir}/{project_name}/features/collaborative_filtering.pkl'
        self.embedding_sim_path = f'{data_dir}/{project_name}/features/embedding_similarity.pkl'

        self.bugid2idx_path = f'{data_dir}/{project_name}/features/report_bugid2idx.pkl'
        self.bugidx2path_idx2valid_path_idx_path = f'{data_dir}/{project_name}/features/bugidx2path_idx2valid_path_idx.pkl'
        self.commit_path2idx_path = f'{data_dir}/{project_name}/features/commit_path2idx.pkl'

        self.dataset_dir = f'{data_dir}/{project_name}/dataset'
        self.svm_result_dir = f'{data_dir}/{project_name}/svm'
