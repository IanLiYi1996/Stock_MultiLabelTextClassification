import sys
sys.path.append('./')

from clean_text import TextProcess, DataClean
from simpletransformers.classification import ClassificationModel
import sklearn
import pandas as pd
import logging, tqdm

procText = TextProcess()
cleaner = DataClean()
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

tag_dict = {'行业/公司基本面-股票投资相关':1,'市场-股票投资相关':1,'技术面-股票投资相关':1, '投资情绪/故事-股票投资相关':1, '房产-其他投资相关':1, '宏观-股票投资相关':1, '其他-其他投资相关':1, '投资理念/策略基本面-股票投资相关':1, '其他-非投资类':0, '基金-其他投资相关':1, '保险-其他投资相关':1, '鸡汤/段子-非投资类':0, '其他-股票投资相关':1, '生活杂事-非投资类':0, '社会事件-非投资类':0, '政治/历史/军事-非投资类':0, '文体八卦-非投资类':0, '社区-非投资类':0}

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
# train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
train_data = []
df = pd.read_csv('/mnt/c/Users/liyi_intern/Documents/Codes/BERT_MLTC/data/cls_data/Mix/train.tsv', sep='\t', usecols=['title','content','tag_name'])
# df = df.loc[:]
for idx, row in tqdm.tqdm(df.iterrows()):
    one_piece = []
    seq = str(row['title']) + str(row['content'])
    if pd.isna(row['title']):
        seq = row['content']
    seq = procText.process_text(cleaner.clean_text(seq))
    one_piece.append(seq)
    one_piece.append(int(tag_dict[row['tag_name']]))
    train_data.append(one_piece)
print(train_data[:10])
train_df = pd.DataFrame(train_data)


# eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
eval_data = []
df = pd.read_csv('/mnt/c/Users/liyi_intern/Documents/Codes/BERT_MLTC/data/cls_data/Mix/test.tsv', sep='\t', usecols=['title','content','tag_name'])
for idx, row in tqdm.tqdm(df.iterrows()):
    one_piece = []
    seq = str(row['title']) + str(row['content'])
    if pd.isna(row['title']):
        seq = row['content']
    seq = procText.process_text(cleaner.clean_text(seq))
    one_piece.append(seq)
    one_piece.append(int(tag_dict[row['tag_name']]))
    eval_data.append(one_piece)
print(eval_data[:10])
eval_df = pd.DataFrame(eval_data)

# # Create a ClassificationModel
model = ClassificationModel('xlnet', '/mnt/c/Users/liyi_intern/Documents/Codes/xlnet_model', use_cuda=False) # You can set class weights by using the optional weight argument

# # Train the model
model.train_model(train_df)

# # Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)