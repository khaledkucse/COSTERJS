import config,logging
import pandas as pd  # For data handling
from logging import config as logging_config
from gensim.scripts.glove2word2vec import glove2word2vec


config.init()
logging.basicConfig(filename=config.log_file_path,
                    filemode='a',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt= '%H:%M:%S',
                    level=logging.INFO)


logging_config.fileConfig(fname='log.conf', defaults={'logfilename': config.log_file_path},disable_existing_loggers=False)
logger = logging.getLogger(__name__)
gloveLogger = logging.getLogger('glove')


def reshapeTrainingDataset(train_dataset):
    for index, row in train_dataset.iterrows():
        tokenPos=int(row['tokenPos'])
        context = row['context']
        label = row['actualLabel']
        tokens = context.strip().split(' ')
        tokens.insert(tokenPos,label)
        train_dataset.at[index,'context'] = ' '.join(tokens)

    train_dataset = train_dataset.drop(train_dataset.columns[[0, 2]], axis=1)
    return train_dataset



def glove():
    print('Semantic Similarity learning by Glove.....')
    gloveLogger.info('Reading train dataset from %s', config.train_dataset_file_path)
    train_dataset = pd.read_csv(config.train_dataset_file_path)
    train_dataset = train_dataset.drop(train_dataset.columns[[0, 1, 2]], axis=1)
    print('Reshaping train dataset.....')
    train_dataset = reshapeTrainingDataset(train_dataset)
    gloveLogger.info('Shape of train dataset: %s', train_dataset.shape)

    gloveLogger.info('Writing the training corpus at %s', config.glove_corpus_file_path)
    print('Writing the training corpus at', config.glove_corpus_file_path)
    filewriter = open(config.glove_corpus_file_path, 'w', encoding='utf-8')
    for index,row in train_dataset.iterrows():
        filewriter.write(str(row['context'])+'\n')
    filewriter.close()

    gloveLogger.info('Training corpus for Glove Training is done.')

glove()

gloveLogger.info('Converting glove result file into word2vec model')
glove2word2vec(config.glove_raw_model_file_path, config.glove_to_word2vec_model_file_path)

gloveLogger.info('Glove model in word2vec format is saced at %s', config.glove_to_word2vec_model_file_path)