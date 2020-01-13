import config,re,logging
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
from logging import config as logging_config
from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec

config.init()
logging.basicConfig(filename=config.log_file_path,
                    filemode='a',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt= '%H:%M:%S',
                    level=logging.INFO)


logging_config.fileConfig(fname='log.conf', defaults={'logfilename': config.log_file_path},disable_existing_loggers=False)
logger = logging.getLogger(__name__)
fastTextLogger = logging.getLogger('fasttext')



class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        fastTextLogger.info('Epoch Completed: {}'.format(self.epoch))
        self.epoch += 1

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)

def getMostFrequentWords(sentences):
    word_freq = defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1
    return word_freq

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


def fastText():
    print('Semantic Similarity learning by FastText.....')
    train_dataset = pd.read_csv(config.train_dataset_file_path)
    train_dataset = train_dataset.drop(train_dataset.columns[[0, 1, 2]], axis=1)
    print('Reshaping train dataset.....')
    train_dataset = reshapeTrainingDataset(train_dataset)
    fastTextLogger.info('Shape of train dataset: %s', train_dataset.shape)

    print('Data Preprocessing.....')
    fastTextLogger.info('Removing the columns with missing context or type....')
    fastTextLogger.info('Train Dataset before cleaning missing data:')
    fastTextLogger.info(train_dataset.isnull().sum())
    train_dataset = train_dataset.dropna().reset_index(drop=True)
    fastTextLogger.info('Train Dataset after cleaning missing data:')
    fastTextLogger.info(train_dataset.isnull().sum())

    fastTextLogger.info('Lemmatizing and non-alphabetic characters for each context')
    #nlp = spacy.load('en', disable=['ner', 'parser'])  # disabling Named Entity Recognition for speed
    brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in train_dataset['context'])

    # t = time()
    # txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]
    # word2vecLogger.info('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    #
    # print(type(txt))
    txt = list(brief_cleaning)

    train_dataset_clean = pd.DataFrame({'cleanContext': txt})
    train_dataset_clean = train_dataset_clean.dropna().drop_duplicates()
    fastTextLogger.info('Shape of cleaned train dataset: %s', train_dataset_clean.shape)

    print('Configuring 3-gram.....')
    context_stream = [row.split() for row in train_dataset_clean['cleanContext']]


    bigram = Phrases(context_stream, min_count=1, delimiter=b' ',progress_per=10000)
    trigram = Phrases(bigram[context_stream], min_count=1, delimiter=b' ', progress_per=10000)


    trigramModel = Phraser(trigram)

    trigrmSentences = trigramModel[context_stream]

    # word_freq = getMostFrequentWords(trigrmSentences)
    # word2vecLogger.info('Number of distinct word in the training dataset: %d',len(word_freq))
    # word2vecLogger.info('Top 10 most frequent words: %s', sorted(word_freq, key=word_freq.get, reverse=True)[:10])

    print('FastText is Learning....')
    cores = multiprocessing.cpu_count()
    fastTextModels = FastText(min_count=5,
                         window=6,
                         size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores - 1,
                         callbacks=[callback()])

    t = time()
    fastTextModels.build_vocab(trigrmSentences, progress_per=10000)
    fastTextLogger.info('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))



    t = time()
    fastTextModels.train(trigrmSentences, total_examples=fastTextModels.corpus_count, epochs=30, report_delay=1)
    fastTextLogger.info('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    #Precompute L2-normalized vectors. The model become read only.
    fastTextModels.wv.init_sims(replace=True)
    print('Saving Word2Vec Modet at',config.fasttext_model_path)
    fastTextLogger.info('Saving Word2Vec Modet at %s', config.fasttext_model_path)
    fastTextModels.wv.save(config.fasttext_model_path,ignore=[])