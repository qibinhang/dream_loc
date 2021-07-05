### How to train
1. run `python prepare_data.py tomcat`
2. run `python code_corpus.py tomcat`
3. run `python vocabulary.py tomcat`
5. run `python dataset.py tomcat`
6. run `python deep_locator.py --project tomcat --batch_size 32`

### How to use pre-trained model
1. run `python deep_locator.py --project tomcat --batch_size 32 --just_test`
