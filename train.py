import config,csv,logging,trainingword2vec,sys,contextsimilarity,trainingfastext
from logging import config as logging_config

config.init()
csv.field_size_limit(sys.maxsize)
logging.basicConfig(filename=config.log_file_path,
                    filemode='a',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt= '%H:%M:%S',
                    level=logging.INFO)


logging_config.fileConfig(fname='log.conf', defaults={'logfilename': config.log_file_path},disable_existing_loggers=False)
train_logger = logging.getLogger('train')



def training(train_files=None):
    #collectTrainDataset(train_files)
    #trainingword2vec.word2vec()
    # contextsimilarity.creteContextDictonary()
    trainingfastext.fastText()



def collectTrainDataset(train_files=None):
    print('Collecting Extracted train data....')
    if train_files == None or len(train_files) == 0:
        train_logger.info('Collecting train files list from %s',config.train_files_list_path)
        train_files = open(config.train_files_list_path, encoding='utf-8').read().strip().split('\n')
        train_logger.info('Total number of training javascript files: %d', len(train_files))

    train_logger.info('Writing Training dataset at %s',config.train_dataset_file_path)
    filewriter = open(config.train_dataset_file_path, 'w', encoding='utf-8')
    filewriter.write(config.dataset_head + '\n')
    filewriter.close()
    no_training_dataset = 0
    count = 0
    for each_file in train_files:
        with open(each_file, newline='') as csvfile:
            reader = csv.reader((line.replace('\0','') for line in csvfile), delimiter=',', quotechar='|')
            lines = list(reader)
        filewriter = open(config.train_dataset_file_path, 'a', encoding='utf-8')
        for each_row in lines:
            if len(each_row) != 6:
                continue
            filewriter.write(', '.join(each_row) + '\n')
            no_training_dataset = no_training_dataset + 1

        filewriter.close()
        count = count+1
        if count%5000 == 0:
            train_logger.info('Data Collection: %d files out of %d are collected.', count,len(train_files))
            train_logger.info('Number of training instances so far collected: %d',no_training_dataset)
    train_logger.info('Total Number of Training Instances: %d', no_training_dataset)