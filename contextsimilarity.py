import config,logging,json
from logging import config as logging_config
import pandas as pd
config.init()
logging.basicConfig(filename=config.log_file_path,
                    filemode='a',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt= '%H:%M:%S',
                    level=logging.INFO)


logging_config.fileConfig(fname='log.conf', defaults={'logfilename': config.log_file_path},disable_existing_loggers=False)
logger = logging.getLogger(__name__)
contextSimilarityLogger = logging.getLogger('contextSimilarity')


def contextPreprocessing(context):
    tokens = context.split(' ')
    filteredcontext = []
    for eachToken in tokens:
        if eachToken not in filteredcontext:
            filteredcontext.append(eachToken)
    return ' '.join(filteredcontext)

def creteContextDictonary():
    print('Constucting Context Dictonary for context similarity calculation.....')
    contextSimilarityLogger.info('Creating Context Dictonary......')
    train_dataset = pd.read_csv(config.train_dataset_file_path)
    train_dataset = train_dataset.drop(train_dataset.columns[[0, 1, 2, 3]], axis=1)
    contextSimilarityLogger.info('Shape of train dataset: %s', train_dataset.shape)

    print('Calculating the frequency of the context for each type/label......')
    contextDictonary = {}
    contextSimilarityLogger.info('Calculating the frequency of the context for each type/label......')
    no_instances = 0
    for index,row in train_dataset.iterrows():
        context = row['context']
        label = row['actualLabel']
        context = contextPreprocessing(context)
        if label in contextDictonary:
            currentvalue = contextDictonary[label]
            if context in currentvalue:
                currentFrequency = currentvalue[context]
                currentvalue[context] = currentFrequency + 1
            else:
                currentvalue[context] = 1

            contextDictonary[label] = currentvalue
        else:
            contextfreq = {context:1}
            contextDictonary[label] = contextfreq
        no_instances = no_instances + 1

        if no_instances%500000 == 0:
            contextSimilarityLogger.info('Data processed: %d instances out of %d.', no_instances, len(train_dataset))

    contextSimilarityLogger.info('Total number of types found in Context Dictonary: %d', len(contextDictonary))

    print('Calculating appearnace score.....')
    contextSimilarityLogger.info('Calculating appearnace score.....')
    no_instances = 0
    finalContextDictonary = {}
    for index,value in contextDictonary.items():
        count = 0
        for context, freq in value.items():
            count = count + freq
        newValue = {}
        for context, freq in value.items():
            if freq > 5:
                newValue[context] = round((freq+0.00001)/(count+0.00001),2)
        if len(newValue)>1:
            finalContextDictonary[index] = newValue
        no_instances = no_instances +1
        if no_instances%1000 == 0:
            contextSimilarityLogger.info('Data processed: %d labels out of %d.', no_instances, len(contextDictonary))

    contextSimilarityLogger.info('Context Dictonary creation completed')
    print('Saving context dictonary at',config.context_dictonary_path)
    contextSimilarityLogger.info('Saving context dictonary at %s',config.context_dictonary_path)
    with open(config.context_dictonary_path, 'w') as fp:
        json.dump(finalContextDictonary, fp)
    