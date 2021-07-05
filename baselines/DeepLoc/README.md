### How to train
1. run `python prepare_data.py tomcat`
2. run `python report_corpus.py tomcat`
3. run `python code_corpus.py tomcat`
4. run `python vocabulary.py tomcat`
5. run `python deep_loc.py --project tomcat --n_kernels 100`

### How to use pre-trained model
1. run `python deep_loc.py --project tomcat --n_kernels 100 --just_test`
