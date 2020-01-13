import config
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

config.init()


def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values
    from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res

le = LabelEncoder()

train_dataset = pd.read_csv(config.train_dataset_file_path)

# Creating train and dev dataframes according to BERT
df_bert = pd.DataFrame(columns=('id', 'label', 'alpha', 'text'))

le.fit(train_dataset['actualLabel'])
encodedLabels = get_integer_mapping(le)


for index, row in train_dataset.iterrows():
    id = row['filePath']+' '+str(row['stmtPos'])+' '+row['codeToken']+' '+str(row['tokenPos'])
    temp_df = pd.DataFrame({'id': id,
                            'label': encodedLabels[row['actualLabel']],
                            'alpha': 'a',
                            'text': row['context']}, index=[index])

    df_bert = df_bert.append(temp_df)

df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.01)

# Creating test dataframe according to BERT
test_dataset = pd.read_csv(config.test_dataset_file_path)
df_bert_test = pd.DataFrame(columns=('id','text'))
for index, row in test_dataset.iterrows():
    id = row['filePath']+' '+str(row['stmtPos'])+' '+row['codeToken']+' '+str(row['tokenPos'])
    temp_df = pd.DataFrame({'id': id,
                            'text': row['context']}, index=[index])

    df_bert_test = df_bert_test.append(temp_df)


# Saving dataframes to .tsv format as required by BERT
df_bert_train.to_csv(config.bert_embedding_data_folder_path+'train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv(config.bert_embedding_data_folder_path+'dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv(config.bert_embedding_data_folder_path+'test.tsv', sep='\t', index=False, header=True)

filewriter = open(config.bert_embedding_data_folder_path+'labeldict.csv', 'w', encoding='utf-8')
filewriter.write('actualLabel,labelID' + '\n')
for id,values in encodedLabels.items():
    filewriter.write(id+','+str(values)+'\n')

filewriter.close()