import sys
sys.path.append('./')

from clean_text import TextProcess, DataClean
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import sklearn
import pandas as pd
import logging, tqdm

procText = TextProcess()
cleaner = DataClean()
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

tag_dict = {'行业/公司基本面-股票投资相关':0,'市场-股票投资相关':1,'技术面-股票投资相关':2, '投资情绪/故事-股票投资相关':3, '房产-其他投资相关':4, '宏观-股票投资相关':5, '其他-其他投资相关':6, '投资理念/策略基本面-股票投资相关':7, '其他-非投资类':8, '基金-其他投资相关':9, '保险-其他投资相关':10, '鸡汤/段子-非投资类':11, '其他-股票投资相关':12, '生活杂事-非投资类':13, '社会事件-非投资类':14, '政治/历史/军事-非投资类':15, '文体八卦-非投资类':16, '社区-非投资类':17}

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
model_args = ClassificationArgs(num_train_epochs=10)
model = ClassificationModel('xlnet', '/mnt/c/Users/liyi_intern/Documents/Codes/xlnet_model', num_labels=18, args=model_args, use_cuda=False) # You can set class weights by using the optional weight argument

# # Train the model
model.train_model(train_df)

# # Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
predictions, raw_outputs = model.predict(["是投资者对董明珠期望太高，还是董小姐做得不好？没有数据就没有真相，没有对比就没有伤害。8年前董明珠从朱洪江手上接过格力的时候，格力空调销售额差不多是美的空调销售额的两倍。8年之后，美的空调销售额超过格力。朱洪江给格力留下的空调霸主地位，被董小姐成功的丢失殆尽。格力现在的困境，其多元化战略失败的根源是董小姐独断专行和缺乏自知自明。明明自己对空调之外的领域一无所知，不招揽良才，非要自己抄刀搞出一出又一出的闹剧。把公司的经营当成表演自己性格和霸气的舞台。如果说多元化的失败只是掌舵人能力的缺失，格力国际化和线上零售的缺席就颇有董小姐私心做崇的嫌疑。作为一个顶级销售，董小姐不可能不知道国际化的重要和线上零售的趋势。然而，格力庞大的国内线下分销体系是董小姐这么多年对抗珠海立于不败的终极武器。国际化和线上零售发展毫无疑问会让董小姐的武器威力大减，在那场权力的游戏中处于不可预知的下风。不知道朱洪江现在看到格力的困境，是否有一丝丝的后悔。"])