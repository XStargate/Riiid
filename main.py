import riiideducation
# import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import gc
from collections import defaultdict
from tqdm.notebook import tqdm
import psutil
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

feld_needed = ['timestamp','user_id','content_id', 'task_container_id','content_type_id', 'answered_correctly', 'prior_question_elapsed_time', 'prior_question_had_explanation']
train = pd.read_pickle('../input/riiid-cross-validation-files/cv1_train.pickle')[feld_needed]
valid = pd.read_pickle('../input/riiid-cross-validation-files/cv1_valid.pickle')[feld_needed]
train = pd.concat([train,valid]).reset_index()
del valid
_ = gc.collect()
question = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
question_cmnts = pd.read_csv('../input/rhhid-question-cmnts/question_cmnts.csv')

train = train.loc[train.content_type_id == False].reset_index(drop=True)
# valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

user_time_last = train[['user_id','timestamp']].groupby('user_id')['timestamp'].last().reset_index()
user_time_last.columns = ['user_id','last_time']
time_last_dict = defaultdict(int)
for cnt,item in enumerate(tqdm(user_time_last[['user_id','last_time']].values)):
    time_last_dict[item[0]] = item[1]/(1000*3600)

train['lag_time'] = train[['user_id','timestamp']].groupby('user_id')['timestamp'].shift()

train['lag_time2'] = train[['user_id','timestamp']].groupby('user_id')['timestamp'].shift(2)

time_mean = 0.009687222222222222
time_mean2 = 0.025640833333333335
time_mean3 = 0.043081666666666664

user_time_last = train[['user_id','lag_time']].groupby('user_id')['lag_time'].max().reset_index()

user_time_last['lag_time'].fillna(-1,inplace=True)

# user_time_last = train[['user_id','lag_time']].groupby('user_id')['lag_time'].max().reset_index()
user_time_last.columns = ['user_id','last_time']
time_last_dict2 = defaultdict(int)
for cnt,item in enumerate(tqdm(user_time_last[['user_id','last_time']].values)):
    time_last_dict2[item[0]] = item[1]/(1000*3600)

user_time_last = train[['user_id','lag_time2']].groupby('user_id')['lag_time2'].max().reset_index()

user_time_last['lag_time2'].fillna(-1,inplace=True)

# user_time_last = train[['user_id','lag_time2']].groupby('user_id')['lag_time2'].max().reset_index()
user_time_last.columns = ['user_id','last_time']
time_last_dict3 = defaultdict(int)
for cnt,item in enumerate(tqdm(user_time_last[['user_id','last_time']].values)):
    time_last_dict3[item[0]] = item[1]/(1000*3600)

del train['lag_time']
del train['lag_time2']
_ = gc.collect()

prior_question_elapsed_time_mean = train.prior_question_elapsed_time.dropna().values.mean()
train['prior_question_elapsed_time_mean'] = train['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean)
user_time_last = train[['user_id','prior_question_elapsed_time_mean']].groupby('user_id')['prior_question_elapsed_time_mean'].last().reset_index()
user_time_last.columns = ['user_id','prior_lag_time']
prior_time_last_dict = defaultdict(int)
for cnt,item in enumerate(tqdm(user_time_last[['user_id','prior_lag_time']].values)):
    prior_time_last_dict[item[0]] = item[1]/(1000*3600)

train['prior_question_had_explanation'].fillna(False,inplace=True)
explanation_agg = train.groupby('user_id')['prior_question_had_explanation'].agg(['sum', 'count'])

explanation_sum_dict = explanation_agg['sum'].astype('int16').to_dict(defaultdict(int))
explanation_count_dict = explanation_agg['count'].astype('int16').to_dict(defaultdict(int))

del explanation_agg
_ = gc.collect()

train["attempt"] = 1
train["attempt"] = train[["user_id","content_id",'attempt']].groupby(["user_id","content_id"])["attempt"].cumsum()

attempt_agg=train.groupby(["user_id","content_id"])["attempt"].agg(['sum'])

attempt_agg=attempt_agg[attempt_agg['sum'] >1]
attempt_agg_dict = attempt_agg['sum'].to_dict(defaultdict(int))

del attempt_agg
_ = gc.collect()

def get_max_attempt(user_id,content_id):
    k = (user_id,content_id)

    if k in attempt_agg_dict.keys():
        attempt_agg_dict[k]+=1
        return attempt_agg_dict[k]

    attempt_agg_dict[k] = 1
    return attempt_agg_dict[k]

target = 'answered_correctly'

user_agg = train.groupby('user_id')[target].agg(['sum', 'count'])

user_sum_dict = user_agg['sum'].astype('int16').to_dict(defaultdict(int))
user_count_dict = user_agg['count'].astype('int16').to_dict(defaultdict(int))
del user_agg
_ = gc.collect()

input_path = '../input/rilld-merge-data/'

answered_correctly_avg_c = pd.read_csv(input_path + 'answered_correctly_avg_c.csv')
answered_correctly_std_c = pd.read_csv(input_path + 'answered_correctly_std_c.csv')
answered_correctly_sum_c = pd.read_csv(input_path + 'answered_correctly_sum_c.csv')
answered_correctly_sum_m = pd.read_csv(input_path + 'answered_correctly_sum_m.csv')
answered_correctly_avg_p = pd.read_csv(input_path + 'answered_correctly_avg_p.csv')
answered_correctly_sum_p = pd.read_csv(input_path + 'answered_correctly_sum_p.csv')
answered_correctly_avg_b = pd.read_csv(input_path + 'answered_correctly_avg_b.csv')
answered_correctly_avg_t = pd.read_csv(input_path + 'answered_correctly_avg_t.csv')
answered_correctly_sum_t = pd.read_csv(input_path + 'answered_correctly_sum_t.csv')

lag_time_avg_b = pd.read_csv(input_path + 'lag_time_avg_b.csv')
lag_time_avg_c = pd.read_csv(input_path + 'lag_time_avg_c.csv')
lag_time_avg_g = pd.read_csv(input_path + 'lag_time_avg_g.csv')
lag_time_avg_m = pd.read_csv(input_path + 'lag_time_avg_m.csv')
lag_time_avg_p = pd.read_csv(input_path + 'lag_time_avg_p.csv')
lag_time_avg_t = pd.read_csv(input_path + 'lag_time_avg_t.csv')

hard_question_sum = pd.read_csv(input_path + 'hard_question_sum.csv')

middle_question_sum = pd.read_csv(input_path + 'middle_question_sum.csv')

easy_question_sum = pd.read_csv(input_path + 'easy_question_sum.csv')
answered_correctly_avg_g = pd.read_csv(input_path + 'answered_correctly_avg_g.csv')
answered_correctly_sum_g = pd.read_csv(input_path + 'answered_correctly_sum_g.csv')
question =pd.read_csv('../input/rilld-merge-data/questions_df.csv')

question.drop('Unnamed: 0',axis=1,inplace=True)

def add_user_feats_without_update(df,user_sum_dict,user_count_dict):
    acsu = np.zeros(len(df), dtype=np.int32)
    cu = np.zeros(len(df), dtype=np.int32)
    for cnt,row in enumerate(df[['user_id']].values):
#         if row[0] in user_sum_dict.keys():
        acsu[cnt] = user_sum_dict[row[0]]
        cu[cnt] = user_count_dict[row[0]]
#         else:
#             acsu[cnt] = 727
#             cu[cnt] = 483
            
    user_feats_df = pd.DataFrame({'user_correct_cumsum':acsu, 'user_correct_cumcount':cu})
    user_feats_df['user_correctness'] = user_feats_df['user_correct_cumsum'] / user_feats_df['user_correct_cumcount']
    df = pd.concat([df, user_feats_df], axis=1)
    return df


def add_explanation(df,explanation_sum_dict,explanation_count_dict):
    acsu_explanation = np.zeros(len(df), dtype=np.int32)
    cu_explanation = np.zeros(len(df), dtype=np.int32)
    for cnt,row in enumerate(df[['user_id','prior_question_had_explanation']].values):
        cu_explanation[cnt] = explanation_count_dict[row[0]] + 1
        acsu_explanation[cnt] = explanation_sum_dict[row[0]] + row[1]
    explanation_feats_df = pd.DataFrame({'explanation_cumsum':acsu_explanation, 'explanation_cumcount':cu_explanation})
    explanation_feats_df['explanation_mean'] = explanation_feats_df['explanation_cumsum'] / explanation_feats_df['explanation_cumcount']
    df = pd.concat([df,explanation_feats_df], axis=1)
    return df


def add_time(df, time_last_dict,time_last_dict2,time_last_dict3):
    time = np.zeros(len(df), dtype=np.float64)
    time2 = np.zeros(len(df), dtype=np.float64)
    time3 = np.zeros(len(df), dtype=np.float64)
    for cnt,row in enumerate(df[['user_id','timestamp']].values):
        if row[0] in time_last_dict.keys():
            time[cnt]=row[1]-time_last_dict[row[0]]
            if(time_last_dict2[row[0]]==time_mean2 or time_last_dict2[row[0]] == -1):#
                time2[cnt]=time_mean2
                time3[cnt]=time_mean3
               
            else:
                time2[cnt]=row[1]-time_last_dict2[row[0]]
                if(time_last_dict3[row[0]]==time_mean3 or time_last_dict3[row[0]] == -1):
                    time3[cnt]=time_mean3 
                else:
                    time3[cnt]=row[1]-time_last_dict3[row[0]]

                time_last_dict3[row[0]]=time_last_dict2[row[0]]

            time_last_dict2[row[0]]=time_last_dict[row[0]]
            time_last_dict[row[0]]=row[1]

        else:
            time[cnt]=time_mean
            time_last_dict[row[0]] = row[1]
            time2[cnt]=time_mean2
            time_last_dict2[row[0]] = time_mean2
            time3[cnt]=time_mean3
            time_last_dict3[row[0]] = time_mean3

    
    time_last_df = pd.DataFrame({'lag_time':time,'lag_time2':time2,'lag_time3':time3})
    
    df = pd.concat([df, time_last_df], axis=1)
    return df



def add_prior_time(df,prior_time_last_dict):
    prior_time = np.zeros(len(df), dtype=np.float64)
    for cnt,row in enumerate(df[['user_id','prior_question_elapsed_time_mean']].values):
        if prior_time_last_dict[row[0]] != 0:
            prior_time[cnt] = row[1] - prior_time_last_dict[row[0]]
            prior_time_last_dict[row[0]] = row[1]
        else:
            prior_time_last_dict[row[0]] = row[1]
            
    time_last_df = pd.DataFrame({'prior_lag_time':prior_time})
    df = pd.concat([df, time_last_df], axis=1)
    return df


def update_user_feats(df, user_sum_dict,user_count_dict):
    temp = []
    for row in df[['user_id','answered_correctly','content_type_id']].values:
        if row[2] == 0:
            user_sum_dict[row[0]] += row[1]
            user_count_dict[row[0]] += 1

del train
_=gc.collect()

user_bundle_first = pd.read_csv('../input/rilld-merge-data/user_bundle_first.csv')
user_tag_first = pd.read_csv('../input/rilld-merge-data/user_tag_first.csv')

user_bundle_first_dict = defaultdict(int)
for cnt,item in enumerate(tqdm(user_bundle_first[['user_id','user_bundle_first']].values)):
    user_bundle_first_dict[item[0]] = item[1]

user_tag_first_dict = defaultdict(int)
for cnt,item in enumerate(tqdm(user_tag_first[['user_id','user_tag_first']].values)):
    user_tag_first_dict[item[0]] = item[1]

def add_first(df,user_bundle_first_dict,user_tag_first_dict):
    temp1 = np.zeros(len(df))
    temp2 = np.zeros(len(df))
    for cnt,row in enumerate(df[['user_id','bundle_id','tags_lsi']].values):
        if row[0] not in user_bundle_first_dict.keys():
            user_bundle_first_dict[row[0]]  = row[1]
            user_tag_first_dict[row[0]]  = row[2]
        temp1[cnt] = user_bundle_first_dict[row[0]]
        temp2[cnt] = user_tag_first_dict[row[0]]
    leature_df = pd.DataFrame({'user_bundle_first':temp1,'user_tag_first':temp2})
    df = pd.concat([df, leature_df], axis=1)
    return df

user_lecture_agg = pd.read_csv('../input/rilld-merge-data/user_lecture_agg.csv')

user_lecture_sum = defaultdict(int)
user_lecture_count = defaultdict(int)
for cnt,item in enumerate(tqdm(user_lecture_agg[['user_id','sum','count']].values)):
    user_lecture_sum[item[0]] = item[1]
    user_lecture_count[item[0]] = item[2]

# hard_question_mean_dict = defaultdict(int)
# for cnt,item in enumerate(tqdm(hard_question_mean[['user_id','hard_question_mean']].values)):
#     hard_question_mean_dict[item[0]] = item[1]
    
hard_question_sum_dict = defaultdict(int)
for cnt,item in enumerate(tqdm(hard_question_sum[['user_id','hard_question_sum']].values)):
    hard_question_sum_dict[item[0]] = item[1]
    
# middle_question_mean_dict = defaultdict(int)
# for cnt,item in enumerate(tqdm(middle_question_mean[['user_id','middle_question_mean']].values)):
#     middle_question_mean_dict[item[0]] = item[1]
    
middle_question_sum_dict = defaultdict(int)
for cnt,item in enumerate(tqdm(middle_question_sum[['user_id','middle_question_sum']].values)):
    middle_question_sum_dict[item[0]] = item[1]
    
# leature_sum_dict = defaultdict(int)
# for cnt,item in enumerate(tqdm(leature_sum[['user_id','leature_sum']].values)):
#     leature_sum_dict[item[0]] = item[1]
    
# leature_mean_dict = defaultdict(int)
# for cnt,item in enumerate(tqdm(leature_mean[['user_id','leature_mean']].values)):
#     leature_mean_dict[item[0]] = item[1]
    
# easy_question_mean_dict = defaultdict(int)
# for cnt,item in enumerate(tqdm(easy_question_mean[['user_id','easy_question_mean']].values)):
#     easy_question_mean_dict[item[0]] = item[1]
    
easy_question_sum_dict = defaultdict(int)
for cnt,item in enumerate(tqdm(easy_question_sum[['user_id','easy_question_sum']].values)):
    easy_question_sum_dict[item[0]] = item[1]

def add_leature(df,user_lecture_sum,user_lecture_count):
    temp1 = np.zeros(len(df))
    temp2 = np.zeros(len(df))
    for cnt,row in enumerate(df[['user_id','content_type_id']].values):
        user_lecture_count[row[0]] = user_lecture_count[row[0]] + 1
        user_lecture_sum[row[0]] = user_lecture_sum[row[0]] +row[1]
        if row[1] == 0:
            temp1[cnt] = user_lecture_sum[row[0]]
            temp2[cnt] =  user_lecture_sum[row[0]]/user_lecture_count[row[0]]
    leature_df = pd.DataFrame({'user_lecture_sum':temp1,'user_lecture_lv':temp2})
    df = pd.concat([df, leature_df], axis=1)
    return df

question_cnt = pd.read_csv('../input/rilld-merge-data/easy_question_count.csv')

question_cnt_dict = defaultdict(int)
for cnt,item in enumerate(tqdm(question_cnt[['user_id','easy_question_count']].values)):
    question_cnt_dict[item[0]] = item[1]

def add_leavel(df,hard_question_sum_dict,middle_question_sum_dict,easy_question_sum_dict,question_cnt_dict):
    easy_sum = np.zeros(len(df))
    easy_mean = np.zeros(len(df))
    middle_sum = np.zeros(len(df))
    middle_mean = np.zeros(len(df))
    hard_sum = np.zeros(len(df))
    hard_mean = np.zeros(len(df))
    for cnt,row in enumerate(df[['user_id','answered_correctly_avg_c']].values):
        question_cnt_dict[row[0]] = question_cnt_dict[row[0]] + 1
        if row[1] < 0.6:
            hard_question_sum_dict[row[0]] = hard_question_sum_dict[row[0]] + 1
        if 0.6<=row[1]<0.83:
            middle_question_sum_dict[row[0]] = middle_question_sum_dict[row[0]] + 1
        if row[1] >= 0.83:
            easy_question_sum_dict[row[0]] = easy_question_sum_dict[row[0]] + 1
        easy_sum[cnt] = easy_question_sum_dict[row[0]]
        easy_mean[cnt] = easy_question_sum_dict[row[0]]/question_cnt_dict[row[0]]
        
        middle_sum[cnt] = middle_question_sum_dict[row[0]]
        middle_mean[cnt] = middle_question_sum_dict[row[0]]/question_cnt_dict[row[0]]
        
        hard_sum[cnt] = hard_question_sum_dict[row[0]]
        hard_mean[cnt] = hard_question_sum_dict[row[0]]/question_cnt_dict[row[0]]
    question_lv = pd.DataFrame({'hard_question_sum':hard_sum,'hard_question_mean':hard_mean,'middle_question_sum':middle_sum,'middle_question_mean':middle_mean,'easy_question_sum':easy_sum,'easy_question_mean':easy_mean})
    df = pd.concat([df, question_lv], axis=1)
    return df

env = riiideducation.make_env()
iter_test = env.iter_test()


MAX_SEQ = 240 # 210
ACCEPTED_USER_CONTENT_SIZE = 2 # 2
EMBED_SIZE = 256 # 256
BATCH_SIZE = 64+32 # 96
DROPOUT = 0.1 # 0.1

class FFN(nn.Module):
    def __init__(self, state_size = 200, forward_expansion = 1, bn_size = MAX_SEQ - 1, dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = state_size
        
        self.lr1 = nn.Linear(state_size, forward_expansion * state_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(bn_size)
        self.lr2 = nn.Linear(forward_expansion * state_size, state_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu(self.lr1(x))
        x = self.bn(x)
        x = self.lr2(x)
        return self.dropout(x)
    
class FFN0(nn.Module):
    def __init__(self, state_size = 200, forward_expansion = 1, bn_size = MAX_SEQ - 1, dropout=0.2):
        super(FFN0, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, forward_expansion * state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(forward_expansion * state_size, state_size)
        self.layer_normal = nn.LayerNorm(state_size) 
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        x=self.layer_normal(x)
        return self.dropout(x)
def future_mask(seq_length):
    future_mask = (np.triu(np.ones([seq_length, seq_length]), k = 1)).astype('bool')
    return torch.from_numpy(future_mask)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads = 8, dropout = DROPOUT, forward_expansion = 1):
        super(TransformerBlock, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, forward_expansion = forward_expansion, dropout=dropout)
        self.ffn0  = FFN0(embed_dim, forward_expansion = forward_expansion, dropout=dropout)
        self.layer_normal_2 = nn.LayerNorm(embed_dim)

    def forward(self, value, key, query, att_mask):
        att_output, att_weight = self.multi_att(value, key, query, attn_mask=att_mask)
        att_output = self.dropout(self.layer_normal(att_output + value))
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        x = self.ffn(att_output)
        x1 = self.ffn0(att_output)
        x = self.dropout(self.layer_normal_2(x + x1 + att_output))
        return x.squeeze(-1), att_weight
    
class Encoder(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, dropout = DROPOUT, forward_expansion = 1, num_layers=1, heads = 8):
        super(Encoder, self).__init__()
        self.n_skill, self.embed_dim = n_skill, embed_dim
        self.embedding = nn.Embedding(2 * n_skill + 1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, forward_expansion = forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, question_ids):
        device = x.device
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        x = self.dropout(x + pos_x)
        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = self.e_embedding(question_ids)
        e = e.permute(1, 0, 2)
        for layer in self.layers:
            att_mask = future_mask(e.size(0)).to(device)
            x, att_weight = layer(e, x, x, att_mask=att_mask)
            x = x.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        return x, att_weight

class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, dropout = DROPOUT, forward_expansion = 1, enc_layers=1, heads = 8):
        super(SAKTModel, self).__init__()
        self.encoder = Encoder(n_skill, max_seq, embed_dim, dropout, forward_expansion, num_layers=enc_layers)
        self.pred = nn.Linear(embed_dim, 1)
        
    def forward(self, x, question_ids):
        x, att_weight = self.encoder(x, question_ids)
        x = self.pred(x)
        return x.squeeze(-1), att_weight
    
def create_model():
    return SAKTModel(n_skill, max_seq=MAX_SEQ, embed_dim=EMBED_SIZE, forward_expansion=1, enc_layers=1, heads=4, dropout=0.1)
skills = joblib.load("../input/riiid-sakt-model-dataset-public/skills.pkl.zip")
n_skill = len(skills)
group = joblib.load("/kaggle/input/newskat/group.pkl.zip")
SATK_model = create_model()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    SATK_model.load_state_dict(torch.load("../input/newskat/sakt_model.pt"))
except:
    SATK_model.load_state_dict(torch.load("../input/newskat/sakt_model.pt", map_location='cpu'))
SATK_model.to(device)
SATK_model.eval()

class TestDataset(Dataset):
    def __init__(self, samples, test_df, skills, max_seq=MAX_SEQ): 
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.skills = skills
        self.n_skill = len(skills)
        self.max_seq = max_seq

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info["user_id"]
        target_id = test_info["content_id"]

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            q_, qa_ = self.samples[user_id]
            
            seq_len = len(q_)

            if seq_len >= self.max_seq:
                q = q_[-self.max_seq:]
                qa = qa_[-self.max_seq:]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_          
        
        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[1:].copy()
        x += (qa[1:] == 1) * self.n_skill
        
        questions = np.append(q[2:], [target_id])
        
        return x, questions

FEATS = ['prior_question_had_explanation',
 'part',
 'tags_num',
 'community',
 'answered_correctly_avg_c',
 'answered_correctly_std_c',
 'answered_correctly_sum_c',
 'answered_correctly_sum_m',
'answered_correctly_avg_b',
'bundle_id',
 'user_correct_cumsum',
 'user_correct_cumcount',
 'user_correctness',
 'attempt',
 'hmean_user_content_accuracy',
 'prior_question_elapsed_time_mean',
 'answered_correctly_avg_p',
 'answered_correctly_sum_p',
 'performance',
 'lag_time',
 'prior_lag_time',
 'explanation_mean',
 'explanation_cumsum',
 'answered_correctly_avg_t',
 'answered_correctly_sum_t',
        'lag_time2','lag_time3','elapsed_time_question',
'user_lecture_sum',
       'user_lecture_lv', 'hard_question_sum', 'middle_question_sum',
       'easy_question_sum', 'hard_question_mean', 'middle_question_mean',
       'easy_question_mean','tags_lsi','answered_correctly_avg_g','answered_correctly_sum_g'
        ,'tag_cor_mean','tag_cor_max','tag_cor_min','user_bundle_first','user_tag_first',
        'lag_time_avg_c', 'lag_time_avg_m', 'lag_time_avg_p', 'lag_time_avg_b',
       'lag_time_avg_t', 'lag_time_avg_g']

import pickle
# file = '../input/riiid-lgb-training-and-save-model/trained_model.pkl'
model = pickle.load(open('../input/rhhid-model/lightgbm_1500_781.pickle', 'rb'))

# current_df = test_df.copy()
# #current_df = data_transform(current_df,False,False)
# current_df['answered_correctly'] = 0
# env.predict(current_df)
# previous_test_df = None

previous_test_df = None
for (test_df, sample_prediction_df) in iter_test:
    if (previous_test_df is not None):
        previous_test_df['answered_correctly'] = eval(test_df["prior_group_answers_correct"].iloc[0])
        ##lightgbm

        update_user_feats(previous_test_df, user_sum_dict,user_count_dict)
        
        #dnn
        previous_test_df = previous_test_df[previous_test_df.content_type_id == False]
        
        prev_group = previous_test_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values))
        for prev_user_id in prev_group.index:
            if prev_user_id in group.index:
                group[prev_user_id] = (
                    np.append(group[prev_user_id][0], prev_group[prev_user_id][0])[-MAX_SEQ:], 
                    np.append(group[prev_user_id][1], prev_group[prev_user_id][1])[-MAX_SEQ:]
                )
 
            else:
                group[prev_user_id] = (
                    prev_group[prev_user_id][0], 
                    prev_group[prev_user_id][1]
                )
            
        
    previous_test_df = test_df.copy()
    

    test_df = add_leature(test_df.reset_index(drop=True),user_lecture_sum,user_lecture_count)
    
    test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop=True)
    ##dnn
    test_dataset = TestDataset(group, test_df, skills)
    test_dataloader = DataLoader(test_dataset, batch_size=51200, shuffle=False)
    
    outs = []

    for item in test_dataloader:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()

        with torch.no_grad():
            output, att_weight = SATK_model(x, target_id)
        outs.extend(torch.sigmoid(output)[:, -1].view(-1).data.cpu().numpy())
    
    ##lightgbm
    test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
    test_df['prior_question_elapsed_time_mean'] = test_df.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
    test_df = add_explanation(test_df,explanation_sum_dict,explanation_count_dict)
    test_df = add_user_feats_without_update(test_df, user_sum_dict, user_count_dict)
    
    test_df = pd.merge(test_df, question, left_on='content_id', right_on='question_id', how='left')
    test_df = pd.merge(test_df, answered_correctly_avg_c, on='content_id',  how="left")
    test_df = pd.merge(test_df, answered_correctly_std_c, on='content_id',  how="left")
    test_df = pd.merge(test_df, answered_correctly_sum_c, on='content_id',  how="left")
    test_df = pd.merge(test_df, answered_correctly_sum_m, on='community',  how="left")
    test_df = pd.merge(test_df, answered_correctly_avg_p, on='part',  how="left")
    test_df = pd.merge(test_df, answered_correctly_sum_p, on='part',  how="left")
    test_df = pd.merge(test_df, answered_correctly_avg_b, on='bundle_id',  how="left")
    test_df = pd.merge(test_df, answered_correctly_avg_t, on='task_container_id',  how="left")
    test_df = pd.merge(test_df, answered_correctly_sum_t, on='task_container_id',  how="left")
    test_df = pd.merge(test_df, answered_correctly_avg_g, on='tags_lsi',  how="left")
    test_df = pd.merge(test_df, answered_correctly_sum_g, on='tags_lsi',  how="left")
    
    test_df = pd.merge(test_df, lag_time_avg_b, on='bundle_id',  how="left")
    test_df = pd.merge(test_df, lag_time_avg_c, on='content_id',  how="left")
    test_df = pd.merge(test_df, lag_time_avg_g, on='tags_lsi',  how="left")
    test_df = pd.merge(test_df, lag_time_avg_m, on='community',  how="left")
    test_df = pd.merge(test_df, lag_time_avg_p, on='part',  how="left")
    test_df = pd.merge(test_df, lag_time_avg_t, on='task_container_id',  how="left")

    test_df = add_first(test_df,user_bundle_first_dict,user_tag_first_dict)
    test_df = add_leavel(test_df,hard_question_sum_dict,middle_question_sum_dict,easy_question_sum_dict,question_cnt_dict)
    test_df['hmean_user_content_accuracy'] = 2 * (
        (test_df['user_correctness'] * test_df['answered_correctly_avg_c']) /
        (test_df['user_correctness'] + test_df['answered_correctly_avg_c'])
    )

    
    
    test_df['performance'] = test_df['answered_correctly_avg_c'] - test_df['user_correctness']
    test_df['timestamp'] = test_df['timestamp']/(1000*3600)
    test_df = add_time(test_df, time_last_dict,time_last_dict2,time_last_dict3)
    
#     test_df = add_user_std(test_df, user_cor_std_dict)
    test_df['prior_question_elapsed_time_mean'] = test_df['prior_question_elapsed_time_mean']/(1000*3600)
    test_df = add_prior_time(test_df,prior_time_last_dict)
    test_df['elapsed_time_question'] = test_df['lag_time'] - test_df['prior_question_elapsed_time_mean'] 
    test_df["attempt"] = test_df[["user_id", "content_id"]].apply(lambda row: get_max_attempt(row["user_id"], row["content_id"]), axis=1)
    ##blending
    test_df['answered_correctly'] = .7*model.predict(test_df[FEATS].values) + .3*np.array(outs)

    env.predict(test_df[['row_id', 'answered_correctly']])
    
model.predict(test_df[FEATS].values)

test_df[['row_id', 'answered_correctly']]
