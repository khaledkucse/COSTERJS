def init():
    global \
        top_k, \
        full_dataset_folder_path,\
        train_dataset_file_path, \
        train_files_list_path, \
        test_dataset_file_path, \
        test_files_list_path, \
        bert_embedding_data_folder_path, \
        glove_corpus_file_path, \
        glove_raw_model_file_path, \
        glove_to_word2vec_model_file_path, \
        result_file_path, \
        log_file_path, \
        output_vocab_file_path, \
        checkpoints_folder_path, \
        word2vec_model_path, \
        fasttext_model_path, \
        context_dictonary_path, \
        type_frequency_dataset_path, \
        dataset_head, \
        result_head,\
        version

    # Predicion parameter
    top_k = 5

    # All File Path
    root_folder = '/home/khaledkucse/Project/backup/costerjs/'

    full_dataset_folder_path = root_folder + 'data/Dataset/'

    train_files_list_path = root_folder +'data/trainfiles.txt'

    train_dataset_file_path = root_folder +'data/train.csv'

    test_files_list_path = root_folder +'data/testfiles.txt'

    test_dataset_file_path = root_folder +'data/test.csv'

    bert_embedding_data_folder_path = root_folder + 'data/bert_embedding/'

    glove_corpus_file_path = root_folder +'data/traincorpus.txt'

    glove_raw_model_file_path = root_folder +'model/gloveresult.txt'

    glove_to_word2vec_model_file_path = root_folder +'model/glovemodel.word2vec'

    result_file_path = root_folder + 'data/results.csv'

    type_frequency_dataset_path = root_folder +'data/typefrequency.csv'

    log_file_path = root_folder+'trainTest.log'

    word2vec_model_path = root_folder +'model/word2vec.model'

    fasttext_model_path = root_folder + 'model/fasttext.model'

    context_dictonary_path = root_folder +'model/contextDictonary.json'

    dataset_head='filePath,'\
                 'stmtPos,' \
                 'codeToken,' \
                 'tokenPos,' \
                 'context,'\
                 'actualLabel'

    result_head = 'filePath,' \
                   'stmtPos,' \
                   'codeToken,' \
                   'tokenPos,' \
                   'context,' \
                   'actualLabel,' \
                   'predictedLabel'
    version='0.1'