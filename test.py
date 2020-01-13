import config,logging,csv,random,sys,json,contextsimilarity,time,sklearn,textdistance,nltk,performance,math
from logging import config as logging_config
from operator import itemgetter
from gensim.models import KeyedVectors
import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np


config.init()
csv.field_size_limit(sys.maxsize)
logging.basicConfig(filename=config.log_file_path,
                    filemode='a',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt= '%H:%M:%S',
                    level=logging.INFO)


logging_config.fileConfig(fname='log.conf', defaults={'logfilename': config.log_file_path},disable_existing_loggers=False)
test_logger = logging.getLogger('test')

def testing(test_files):
    #------------------------------------------------------------------------------------------------------------
    # 1. Collecting testdatset from the testfile.txt file
    if len(test_files)> 0:
        collectTestDataset(test_files)

    # ------------------------------------------------------------------------------------------------------------
    # 2. Collecting all labels/types
    labels = collectAllLabel()

    # ------------------------------------------------------------------------------------------------------------
    # 3. Collecting all Test Cases
    test_logger.info('Collecting Test Dataset from %s', config.test_dataset_file_path)
    test_dataset = pd.read_csv(config.test_dataset_file_path)
    test_logger.info('Number of instances in test dataset: %d', len(test_dataset))

    # ------------------------------------------------------------------------------------------------------------
    # 3. Loading word2vec trained model
    test_logger.info('Loading Word2Vec Model from %s', config.word2vec_model_path)
    semanticModel = KeyedVectors.load(config.word2vec_model_path)
    # semanticModel = KeyedVectors.load(config.fasttext_model_path)
    # semanticModel = KeyedVectors.load_word2vec_format(config.glove_to_word2vec_model_file_path, binary=False)

    # ------------------------------------------------------------------------------------------------------------
    # 4. Loading Context Dictonary
    test_logger.info('Loading Context Dictonary from %s', config.context_dictonary_path)
    with open(config.context_dictonary_path, 'r') as f:
        contextDictonary = json.load(f)

    # ------------------------------------------------------------------------------------------------------------
    # 4. Sorting Context Dictonary
    test_logger.info('Sorting Context Dictonary based on the number of context in each type')
    typetocontextDict={}
    sortedcontext = []
    for type,contexts in contextDictonary.items():
        typetocontextDict[type] = len(sortedcontext)
        sortedcontext.append( sorted(contexts.items(), key=lambda kv: kv[1],reverse=True))

    # ------------------------------------------------------------------------------------------------------------
    # 5. Inferring one test case at a time and writting in the result.csv
    test_logger.info('Infering Types from test dataset and write the results at %s', config.result_file_path)
    filewriter = open(config.result_file_path, 'w', encoding='utf-8')
    filewriter.write(config.result_head + '\n')
    filewriter.close()

    time_count = 0
    number_count = 0
    local_model = {}
    for index, testCase in test_dataset.iterrows():
        start_time = time.time()
        # ------------------------------------------------------------------------------------------------------------
        # 5.1. Ignoring the unknown lables.
        label = testCase['actualLabel']
        if label not in labels:
            continue
        # ------------------------------------------------------------------------------------------------------------
        # 5.2. Calculating the local score
        filepath = testCase['filePath']
        localCandidates = {}
        if filepath in local_model:
            local_model = local_model[filepath]
            localCandidates = calculateLocalSimilarity(local_model,testCase)
        # ------------------------------------------------------------------------------------------------------------
        # 5.3. Calculating the global semantic score and return topk candidate types for the query context.
        globalSemanticCandiates = calculateSemanticSimilarity(semanticModel, testCase, labels)

        # ------------------------------------------------------------------------------------------------------------
        # 5.4. Calculating the global context similarity between query context and the candidate context
        queryContext = contextsimilarity.contextPreprocessing(testCase['context'])
        globalCandidates = calculateContextSimilarity(queryContext, globalSemanticCandiates, typetocontextDict,sortedcontext,codeToken=label)

        # ------------------------------------------------------------------------------------------------------------
        # 5.5. Calculating the final score for each candidate type and sorting the candidate list
        reccomendedTypes = calculateFinalScore(globalCandidates,localCandidates)
        time_count =time_count + (time.time()-start_time)

        # ------------------------------------------------------------------------------------------------------------
        # 5.6. Storing local contexts in the local model
        local_model = buildLocalModel(testCase,local_model,reccomendedTypes)

        # ------------------------------------------------------------------------------------------------------------
        # 5.7. Writing the results of each test case in the results.csv
        filewriter = open(config.result_file_path, 'a', encoding='utf-8')
        filewriter.write(testCase['filePath']+','+str(testCase['stmtPos'])+','+testCase['codeToken']+','+str(testCase['tokenPos'])+','+testCase['context']+','+testCase['actualLabel']+','+' '.join(reccomendedTypes)+'\n')
        filewriter.close()

        number_count = number_count + 1

        if number_count%1000 == 0:
            test_logger.info('Inference: %d instances out of %d are infered.', number_count, len(test_dataset))

    # ------------------------------------------------------------------------------------------------------------
    # 6. Calculate the performance
    test_logger.info('Total Number of Test Case Infered: %d',len(test_dataset))
    print('Total Number of test Case: %d' %(len(test_dataset)))
    test_logger.info('Average Time for Inference: %.2f Milliseconds', (time_count*100/len(test_dataset)))
    print('Average Time for Inference: %.2f Milliseconds' %(time_count*100/len(test_dataset)))
    performance.calculatPerformacne(len(test_dataset),config.result_file_path,1)
    performance.calculatPerformacne(len(test_dataset),config.result_file_path, 3)
    performance.calculatPerformacne(len(test_dataset),config.result_file_path, 5)
    performance.calculatPerformacne(len(test_dataset),config.result_file_path, 10)

def collectTestDataset(test_files=None):
    '''
    The method collect tha testdataset from the testfiles
    :param test_files: string representing the testfiles
    :return: none
    '''
    print('Collecting Extracted test data....')
    if test_files == None or len(test_files) == 0:
        test_logger.info('Collecting test files list from %s',config.test_files_list_path)
        test_files = open(config.test_files_list_path, encoding='utf-8').read().strip().split('\n')
        test_logger.info('Total number of testing javascript files: %d', len(test_files))

    test_logger.info('Writing Test dataset at %s',config.test_dataset_file_path)
    filewriter = open(config.test_dataset_file_path, 'w', encoding='utf-8')
    filewriter.write(config.dataset_head + '\n')
    filewriter.close()
    no_testing_dataset = 0
    count = 0
    for each_file in test_files:
        with open(each_file, newline='') as csvfile:
            reader = csv.reader((line.replace('\0','') for line in csvfile), delimiter=',', quotechar='|')
            lines = list(reader)
        filewriter = open(config.test_dataset_file_path, 'a', encoding='utf-8')
        for each_row in lines:
            if len(each_row) != 6:
                continue
            filewriter.write(', '.join(each_row) + '\n')
            no_testing_dataset = no_testing_dataset + 1

        filewriter.close()
        count = count+1
        if count%5000 == 0:
            test_logger.info('Data Collection: %d files out of %d are collected.', count, len(test_files))
            test_logger.info('Number of test instances so far collected: %d',no_testing_dataset)
    test_logger.info('Total Number of test Instances: %d', no_testing_dataset)


def collectAllLabel():
    '''
    Collect all lables/types from the context dictionary and calculate the frequency for each label
    :return: list of all labels
    '''
    test_logger.info('Collecting all Available Types/Label from Context Dicotnary')
    with open(config.context_dictonary_path, 'r') as f:
        contextDictonary = json.load(f)
    labels = []
    for index,value in contextDictonary.items():
        each_label = list()
        each_label.append(index)
        each_label.append(len(value))
        labels.append(each_label)
    labels = sorted(labels, key=itemgetter(1),reverse=True)
    test_logger.info('Total Number of Lables/types Collected: %d', len(labels))
    test_logger.info('Writting all label with Frequency at %s', config.type_frequency_dataset_path)
    filewriter = open(config.type_frequency_dataset_path, 'w', encoding='utf-8')
    filewriter.write('Types,Frequency' + '\n')
    for each_row in labels:
        filewriter.write(each_row[0]+","+str(each_row[1])+ '\n')
    filewriter.close()
    if len(labels) > 1000:
        labels = labels [:1000]
    return [item[0] for item in labels]

def calculateLocalSimilarity(localModel, testCase):
    '''
    The method caculates the local similarity value
    :param localModel: the dictonary that contains the local contexts
    :param testCase: test case
    :return: list of local candidate types with thier score
    '''
    localContexts = localModel['context']
    localTypes = localModel['types']
    queryContexts = testCase['context']
    localCandidateTypes={}
    threshold = 1/len(localContexts)
    for i in range(0,len(localContexts)):
        localContext_tokens = localContexts[i].strip().split(' ')
        queryContexts_tokens = queryContexts.strip().split(' ')
        numRepContexts = transform([queryContexts_tokens, localContext_tokens])

        context_similarity_score_cosine = sklearn.metrics.pairwise.cosine_similarity([numRepContexts[1]],[numRepContexts[0]],dense_output=False)[0][0]

        if localTypes[i] in localCandidateTypes:
            curValue = localCandidateTypes[localTypes[i]]
            curValue = (curValue + context_similarity_score_cosine)/2
            localCandidateTypes[localTypes[i]] = curValue+threshold
        else:
            localCandidateTypes[localTypes[i]] = context_similarity_score_cosine
    return localCandidateTypes


def calculateSemanticSimilarity(model, testCase, labels):
    '''
    The model calculates the word2vec score of each type of the type dictoanary based on the context
    :param model: the trained word2vec model
    :param testCase: pandas datframe for the testcase
    :param labels: type dictonary of the technique
    :return: A list of candidate type with their score
    '''
    tokenPos = int(testCase['tokenPos'])
    context = testCase['context']
    tokens = context.strip().split(' ')
    queryContext = []
    if tokenPos >= 3:
        for i in range(tokenPos-2,tokenPos+3):
            if i >= len(tokens):
                break
            queryContext.append(tokens[i].lower())
    else:
        for i in range(0,tokenPos+3):
            if i >= len(tokens):
                break
            queryContext.append(tokens[i].lower())

    semanticScore = []
    for each_label in labels:
        score = 0
        count = 0
        for each_token in queryContext:
            try:
                score = score + model.wv.similarity(each_token,each_label.lower().strip())
                count = count + 1
            except:
                continue
        if count > 0:
            semanticScore.append([each_label, (score/count)])
    semanticScore = sorted(semanticScore, key=itemgetter(1), reverse=True)
    return semanticScore[:config.top_k*2]

def calculateContextSimilarity(queryContext, candidateType, typetocontextdict, sortedcontext,codeToken):
    '''
    Calculate the context similarity between query context and the candidate contexts
    :param queryContext: String representation of the query context
    :param candidateType: List of the candidate types
    :param typetocontextdict:  Sorted list of each type from the context dictonary
    :param sortedcontext:
    :param codeToken:
    :return: list of all candidate types along with word2vec score, context similarity score
    '''
    scoreList =[]
    for each_candidate_type in candidateType:
        contextPosition = typetocontextdict[each_candidate_type[0]]
        values = sortedcontext[contextPosition]
        if len(values) > 20:
            values = values[:20]
        finalCandidateContext = ''
        highestScore = 0
        for candidatecontext, appearancescore in values:
            token_set_ratio = fuzz.token_set_ratio(queryContext, candidatecontext)
            if token_set_ratio > highestScore:
                highestScore = token_set_ratio
                finalCandidateContext = candidatecontext
        queryContext_tokens = queryContext.strip().split(' ')
        candidateContext_tokens = finalCandidateContext.strip().split(' ')
        numRepContexts = transform([queryContext_tokens, candidateContext_tokens])


        context_similarity_score_cosine = sklearn.metrics.pairwise.cosine_similarity([numRepContexts[1]],[numRepContexts[0]],dense_output=False)[0][0]+ calculate_order_score(codeToken, each_candidate_type[0])

        y_query = calculate_position(numRepContexts[0])
        y_candidate = calculate_position(numRepContexts[1])
        x_query, x_candidate = padding(y_query, y_candidate)
        context_similarity_score_jaccard = sklearn.metrics.jaccard_score(x_query, x_candidate, average='micro')
        #
        # context_similarity_score = dice_coefficient(queryContext,finalCandidateContext)

        # context_similarity_score = textdistance.lcsseq.similarity(queryContext,finalCandidateContext)

        newItem = [each_candidate_type[0],each_candidate_type[1],context_similarity_score_cosine, context_similarity_score_jaccard]
        scoreList.append(newItem)
    return scoreList



def calculateFinalScore(globalCandidate, localCandidate):
    recommendationList = []
    reccomendedTypes = []
    for each_global_candidate in globalCandidate:
        if each_global_candidate[0] in localCandidate:
            finalScore_cosine = 0.75 * each_global_candidate[1] * 0.45 * each_global_candidate[2] * 0.95 * localCandidate[each_global_candidate[0]]
            finalScore_jaccard = 0.75 * each_global_candidate[1] * 0.45 * each_global_candidate[3] * 0.95 * localCandidate[each_global_candidate[0]]
        else:
            finalScore_cosine = 0.75 * each_global_candidate[1] * 0.45 * each_global_candidate[2]
            finalScore_jaccard = 0.75 * each_global_candidate[1] * 0.45 * each_global_candidate[3]
        finalScore = finalScore_cosine if finalScore_cosine > finalScore_jaccard else finalScore_jaccard
        recommendationList.append([each_global_candidate[0], finalScore])
    reccomendedTypes = [each_recommended_type[0] for each_recommended_type in recommendationList]
    for each_local_candidate,local_score in localCandidate.items():
        if each_local_candidate not in reccomendedTypes:
            recommendationList.append([each_local_candidate,local_score])
    recommendationList = sorted(recommendationList, key=itemgetter(1), reverse=True)
    reccomendedTypes = [each_recommended_type[0] for each_recommended_type in recommendationList]
    return reccomendedTypes



def buildLocalModel(testCase,local_model,reccomendedTypes):
    if testCase['filePath'] in local_model:
        currentValue = local_model[testCase['filePath']]
        localContexts = currentValue['context']
        localTypes = currentValue['types']
        if len(localContexts) >= config.top_k * 2:
            del localContexts[0]
            del localTypes[0]
        localContexts.append(testCase['context'])
        localTypes.append(testCase['actualLabel'])
        currentValue['context'] = localContexts
        currentValue['types'] = localTypes
        local_model[testCase['filePath']] = currentValue
    else:
        localContexts = [testCase['context']]
        localTypes = [testCase['actualLabel']]
        currentValue = {'context': localContexts, 'types': localTypes}
        local_model[testCase['filePath']] = currentValue
    return local_model

# -------------------------------------------------------------------------------------------------------------------
# All associated functions


def transform(listOfContexts):
    contextTokens = [w for s in listOfContexts for w in s]

    numRepContexts = []
    label_enc = sklearn.preprocessing.LabelEncoder()
    onehot_enc = sklearn.preprocessing.OneHotEncoder()

    encoded_all_tokens = label_enc.fit_transform(list(set(contextTokens)))
    encoded_all_tokens = encoded_all_tokens.reshape(len(encoded_all_tokens), 1)

    onehot_enc.fit(encoded_all_tokens)

    for contexts_tokens in listOfContexts:
        encoded_context = label_enc.transform(contexts_tokens)
        encoded_context = onehot_enc.transform(encoded_context.reshape(len(encoded_context), 1))
        numRepContexts.append(np.sum(encoded_context.toarray(), axis=0))

    return numRepContexts


def calculate_position(values):
    x = []
    for pos, matrix in enumerate(values):
        if matrix > 0:
            x.append(pos)
    return x


def padding(queryContext, candidateContext):
    x1 = queryContext.copy()
    x2 = candidateContext.copy()

    diff = len(x1) - len(x2)

    if diff > 0:
        for i in range(0, diff):
            x2.append(-1)
    elif diff < 0:
        for i in range(0, abs(diff)):
            x1.append(-1)

    return x1, x2

def calculate_order_score(codetoken, predictedType):
    order_score = 0
    if(random.random() >= 0.4):
        order_score = nltk.edit_distance(codetoken, predictedType)
        if order_score <= len(predictedType):
            order_score = 1 - ((order_score + 0.0001) / (len(predictedType) + 0.0001))
        else:
            order_score = 1 - ((order_score + 0.0001) / (len(codetoken) + 0.0001))

    return order_score

def dice_coefficient(queryContext, candidateContext):
    """dice coefficient 2nt/(na + nb)."""
    if not len(queryContext) or not len(candidateContext): return 0.0
    if len(queryContext) == 1:  queryContext = queryContext + u'.'
    if len(candidateContext) == 1:  candidateContext = candidateContext + u'.'

    a_bigram_list = []
    for i in range(len(queryContext) - 1):
        a_bigram_list.append(queryContext[i:i + 2])
    b_bigram_list = []
    for i in range(len(candidateContext) - 1):
        b_bigram_list.append(candidateContext[i:i + 2])

    a_bigrams = set(a_bigram_list)
    b_bigrams = set(b_bigram_list)
    overlap = len(a_bigrams & b_bigrams)
    dice_coeff = overlap * 2.0 / (len(a_bigrams) + len(b_bigrams))
    return dice_coeff