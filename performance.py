import config,logging
from logging import config as logging_config
import pandas as pd

config.init()
logging.basicConfig(filename=config.log_file_path,
                    filemode='a',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt= '%H:%M:%S',
                    level=logging.INFO)


logging_config.fileConfig(fname='log.conf', defaults={'logfilename': config.log_file_path},disable_existing_loggers=False)
test_logger = logging.getLogger('test')


def isCorrectlyPredicted(actualLabel,predictionLabel):
    for each_prediction in predictionLabel:
        if each_prediction.strip() == '':
            continue
        if each_prediction.strip().lower().replace('[]','') == actualLabel.strip().lower().replace('[]',''):
            return 1

    return 0


def calculatPerformacne(total_test_case,result_file_path = config.result_file_path,top_k=5):
    results = pd.read_csv(result_file_path)
    true_positive = 0
    false_positive = 0
    for index, row in results.iterrows():
        actualLabel = row['actualLabel']
        predictedLabel = row['predictedLabel']
        predictedTypes = predictedLabel.strip().split(' ')
        predictedTypes = [i for i in predictedTypes if i.strip() != '']

        if isCorrectlyPredicted(actualLabel,predictedTypes[:top_k]):
            true_positive = true_positive + 1
        else:
            false_positive = false_positive + 1

    precision = (true_positive+0.001)/(true_positive+false_positive+0.0001)
    recall = (true_positive+0.0001)/(total_test_case+0.0001)
    fscore = (2*precision*recall)/(precision+recall)

    print('Top-%s Recommendation:' %(top_k))
    test_logger.info('Top-%s Recommendation:', top_k)
    print('Precision: %0.2f' %(precision*100))
    test_logger.info('Precision: %0.2f',(precision*100))
    print('Recall: %0.2f' %(recall*100))
    test_logger.info('Recall: %0.2f', (recall*100))
    print('F1 Score: %0.2f' %(fscore*100))
    test_logger.info('F1 Score: %0.2f', (fscore*100))
    print('------------------------------')
    test_logger.info('------------------------------')
