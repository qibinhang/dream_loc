import sys
from Corpus.code_collector import main as code_collector_main
from Corpus.corpus import main as corpus_main
from Features.feature import main as feature_main
from vocabulary import main as vocab_main
from dataset import main as dataset_main


if __name__ == '__main__':
    project_name = sys.argv[1]
    assert project_name in ('swt', 'tomcat', 'birt', 'eclipse', 'jdt')
    code_collector_main(project_name)
    corpus_main(project_name)
    feature_main(project_name)
    vocab_main(project_name)
    dataset_main(project_name)
