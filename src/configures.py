class Configs:
    def __init__(self, project_name):
        project_id = {
            'swt': 0, 'tomcat': 1, 'birt': 2, 'eclipse': 3, 'jdt': 4
        }

        assert project_name in project_id
        data_dir = '../data'
        projects = [('SWT.xml', 'eclipse.platform.swt'),
                    ('Tomcat.xml', 'tomcat'),
                    ('Birt.xml', 'birt'),
                    ('Eclipse_Platform_UI.xml', 'eclipse.platform.ui'),
                    ('JDT.xml', 'eclipse.jdt.ui')]
        p_id = project_id[project_name]
        r_name, c_name = projects[p_id]
        self.data_dir = data_dir
        self.r_name = r_name
        self.c_name = c_name
        self.data_dir = data_dir
        self.report_path = f'{data_dir}/reports/{r_name}'
        self.code_dir = f'{data_dir}/source_code/{c_name}'

        # for efficiency, 200 for birt, eclipse, JDT, and SWT, 800 for tomcat
        if project_name == 'swt':
            self.len_code_snippet = 100
            self.max_num_snippet = 20
            self.num_neg_sample = 200
        elif project_name == 'tomcat':
            self.len_code_snippet = 100
            self.max_num_snippet = 20
            self.num_neg_sample = 800
        elif project_name == 'birt':
            self.len_code_snippet = 100
            self.max_num_snippet = 20
            self.num_neg_sample = 200
        elif project_name == 'eclipse':
            self.len_code_snippet = 100
            self.max_num_snippet = 20
            self.num_neg_sample = 200
        elif project_name == 'jdt':
            self.len_code_snippet = 100
            self.max_num_snippet = 20
            self.num_neg_sample = 200

        complete_c_name = project_name

        self.corpus_dir = f'{data_dir}/{complete_c_name}/corpus'
        self.feature_dir = f'{data_dir}/{complete_c_name}/features'
        self.vocab_dir = f'{data_dir}/{complete_c_name}/vocabulary'
        self.dataset_dir = f'{data_dir}/{complete_c_name}/dataset'
        self.model_log_dir = f'{data_dir}/{complete_c_name}/log'
        self.model_dir = f'{data_dir}/{complete_c_name}/models'

        self.max_len_report = 300
        self.num_cpu = 20

    def modify(self, len_snippet, num_snippet):
        self.len_code_snippet = len_snippet
        self.max_num_snippet = num_snippet

        complete_c_name = f'{self.c_name}_snippet_{self.len_code_snippet}_neg_{self.num_neg_sample}'

        self.corpus_dir = f'{self.data_dir}/{complete_c_name}/corpus'
        self.feature_dir = f'{self.data_dir}/{complete_c_name}/features'
        self.vocab_dir = f'{self.data_dir}/{complete_c_name}/vocabulary'
        self.dataset_dir = f'{self.data_dir}/{complete_c_name}/dataset'
        self.model_log_dir = f'{self.data_dir}/{complete_c_name}/log'
        self.model_dir = f'{self.data_dir}/{complete_c_name}/models'
