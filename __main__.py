import config,os,logging,train,test,warnings
from glob import glob as glob
from logging import config as logging_config
from sklearn.model_selection import train_test_split

config.init()
warnings.filterwarnings('ignore')

logging.basicConfig(filename=config.log_file_path,
                    filemode='a',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt= '%H:%M:%S',
                    level=logging.INFO)


logging_config.fileConfig(fname='log.conf', defaults={'logfilename': config.log_file_path},disable_existing_loggers=False)
logger = logging.getLogger(__name__)




print('COSTERJS: A Fast and Scalable Java Script Type Infer Tool')
print('Version:'+config.version)
print('\nTrain and Test:\n=============================================================================================')
print('To see the more information for each project open '+config.log_file_path+'\n\n\n')
train_files=[]
test_files=[]
if not os.path.isfile(config.train_files_list_path) or not os.path.isfile(config.test_files_list_path):
    print('Collecting extracted java script source files and splitting them into traina nd test.....')
    logger.info('Collecting projects.....')
    files = [y for x in os.walk(config.full_dataset_folder_path) for y in glob(os.path.join(x[0], '*.csv'))]
    logger.info('Total Scripts Files: %d',len(files))
    logger.info('Spliting source files into train and test by 90-10 percentage')
    train_files, test_files = train_test_split(files, test_size=0.1)

    logger.info('Total number of training files: %d', len(train_files))
    logger.info('Total number of testing files: %d', len(test_files))

    logger.info('Writing train source files list at %s', config.train_files_list_path)
    filewriter = open(config.train_files_list_path, 'w', encoding='utf-8')
    for each_file in train_files:
        filewriter.write(each_file + '\n')
    filewriter.close()

    logger.info('Writing test source files list at %s', config.test_files_list_path)
    filewriter = open(config.test_files_list_path, 'w', encoding='utf-8')
    for each_file in test_files:
        filewriter.write(each_file + '\n')
    filewriter.close()

var = input("Enter one of the modes: \n"
            " train : To train the model \n"
            " test: To test the model \n"
            " train-test: To train and then test \n"
            " infer: To see result for a single instance \n")

if var == 'train':
    train.training(train_files)

if var == 'test':
    test.testing(test_files)





