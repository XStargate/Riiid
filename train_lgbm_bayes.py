import pandas as pd
import numpy as np
import gc
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm.notebook import tqdm
import lightgbm as lgb
from os.path import join

data_dir = '../../data_bayes'
save_path = './'
load_path = '../../data_bayes'

train_all = pd.read_csv(join(data_dir, 'train.csv'))

feld_needed = ['timestamp','user_id','content_id','task_container_id', 'content_type_id', 'answered_correctly', 'prior_question_elapsed_time', 'prior_question_had_explanation']
train_all = train_all[feld_needed]

target = 'answered_correctly'
cum = train_all.groupby('user_id')['content_type_id'].agg(['cumsum', 'cumcount'])
cum['cumcount']=cum['cumcount']+1
train_all['user_lecture_sum'] = cum['cumsum'] 
train_all['user_lecture_lv'] = cum['cumsum'] / cum['cumcount']

train_all = train_all.loc[train_all.content_type_id == False].reset_index(drop=True)
train_all['timestamp_lag2'] = train_all[['user_id', 'timestamp']].groupby(['user_id'])[['timestamp']].shift(2)
train_all['lag_time2'] = train_all['timestamp'] - train_all['timestamp_lag2']
train_all['timestamp_lag3'] = train_all[['user_id', 'timestamp']].groupby(['user_id'])[['timestamp']].shift(3)
train_all['lag_time3'] = train_all['timestamp'] - train_all['timestamp_lag3']
train_all['lag_time'] = train_all[['user_id', 'timestamp']].groupby(['user_id'])['timestamp'].diff()

train_all['lag_time'] = train_all['lag_time']/(1000*3600)
train_all['lag_time2'] = train_all['lag_time2']/(1000*3600)
train_all['lag_time3'] = train_all['lag_time3']/(1000*3600)

time_mean = train_all['lag_time'].median()
time_mean2 = train_all['lag_time2'].median()
time_mean3 = train_all['lag_time3'].median()

print (time_mean, time_mean2, time_mean3)

train_all['lag_time'].fillna(time_mean,inplace=True)
train_all['lag_time2'].fillna(time_mean2,inplace=True)
train_all['lag_time3'].fillna(time_mean3,inplace=True)

prior_question_elapsed_time_mean = train_all.prior_question_elapsed_time.dropna().values.mean()
train_all['prior_question_elapsed_time_mean'] = train_all.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
# valid['prior_question_elapsed_time_mean'] = valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
train_all['prior_lag_time'] = train_all[['user_id', 'prior_question_elapsed_time_mean']].groupby(['user_id'])['prior_question_elapsed_time_mean'].diff()

train_all['prior_question_elapsed_time_mean'] = train_all['prior_question_elapsed_time_mean']/(1000*3600)
train_all['prior_lag_time'] = train_all['prior_lag_time']/(1000*3600)

train_all['elapsed_time_question'] = train_all['lag_time'] - train_all['prior_question_elapsed_time_mean']
train_all['prior_lag_time'].fillna(train_all['prior_lag_time'].mean(),inplace=True)
train_all['prior_question_had_explanation'].fillna(False, inplace=True)
train_all['prior_question_had_explanation'] = train_all['prior_question_had_explanation'].astype('int')

train_all['lag'] = train_all[['user_id',target]].groupby('user_id')[target].shift()
cum = train_all[['user_id','lag']].groupby('user_id')['lag'].agg(['cumsum', 'cumcount'])
train_all['user_correctness'] = cum['cumsum'] / cum['cumcount']
train_all['user_correct_cumsum'] = cum['cumsum']
train_all['user_correct_cumcount'] = cum['cumcount']
train_all.drop(columns=['lag'], inplace=True)
train_all['lag'] = train_all.groupby('user_id')['prior_question_had_explanation'].shift()

cum = train_all.groupby('user_id')['lag'].agg(['cumsum', 'cumcount'])
train_all['explanation_mean'] = cum['cumsum'] / cum['cumcount']
train_all['explanation_cumsum'] = cum['cumsum'] 
train_all.drop(columns=['lag'], inplace=True)

train_all["attempt"] = 1
train_all["attempt"] = train_all[["user_id","content_id",'attempt']].groupby(["user_id","content_id"])["attempt"].cumsum()

debug = False
if debug:
    train_all = train_all[:1000000]
    train = train_all[:-250000]
    valid = train_all[-250000:]
else:
    train = train_all

questions_df = pd.read_csv(join(data_dir, 'questions.csv'))
question_cmnts = pd.read_csv(join(data_dir, 'question_cmnts.csv'))
question_tag = pd.read_csv(join(data_dir, 'question_tag.csv'))

index_tags = questions_df['tags'].str.split(' ').explode().reset_index()
index_tags['tags_num'] = index_tags.groupby('index')['tags'].transform('nunique')
index_tags = index_tags.drop_duplicates('index')

questions_df = questions_df.merge(index_tags[['index','tags_num']],left_on='question_id',right_on='index',how='left')
questions_df['tags_num'] = questions_df['tags_num'].apply(lambda x : 1 if x == 0 else x)
questions_df = questions_df.merge(question_cmnts,how='left',left_on='question_id',right_on='Unnamed: 0')
questions_df['community'] = questions_df['community'].apply(lambda x: 2 if x == 4 else x)
questions_df = questions_df.merge(question_tag,how='left',on='question_id')


question_cor = pd.read_csv(join(data_dir, 'tag_cor.csv'))
index_tags = questions_df['tags'].str.split(' ').explode().reset_index()
index_tags.dropna(inplace = True)
index_tags['tags'] = index_tags['tags'].astype('int64')
index_tags = index_tags.merge(question_cor,on='tags',how='left')
index_tags['tag_cor_mean'] = index_tags.groupby('index')['answered_correctly'].transform('mean')
index_tags['tag_cor_max'] = index_tags.groupby('index')['answered_correctly'].transform('max')
index_tags['tag_cor_min'] = index_tags.groupby('index')['answered_correctly'].transform('min')
index_tags = index_tags.drop_duplicates('index')

questions_df = questions_df.merge(index_tags[['index','tag_cor_mean','tag_cor_max','tag_cor_min']],on='index',how='left')
train = pd.merge(train, questions_df[['question_id', 'part','tags_num','community','bundle_id','tags_lsi','tag_cor_mean','tag_cor_max','tag_cor_min']], left_on = 'content_id', right_on = 'question_id', how = 'left')
# valid = pd.merge(valid, questions_df[['question_id', 'part','tags_num','community','bundle_id']], left_on = 'content_id', right_on = 'question_id', how = 'left')
train['user_bundle_first'] = train[['user_id','bundle_id']].groupby('user_id')['bundle_id'].transform('first')
train['user_tag_first'] = train[['user_id','tags_lsi']].groupby('user_id')['tags_lsi'].transform('first')
train[['user_id','user_bundle_first']].drop_duplicates().to_csv('user_bundle_first.csv',index=False)
train[['user_id','user_tag_first']].drop_duplicates().to_csv('user_tag_first.csv',index=False)

answered_correctly_avg_c = train[['content_id','answered_correctly']].groupby(['content_id']).agg('mean').reset_index()
answered_correctly_avg_c.columns = ['content_id','answered_correctly_avg_c']
answered_correctly_std_c = train[['content_id','answered_correctly']].groupby(['content_id']).agg('std').reset_index()
answered_correctly_std_c.columns = ['content_id','answered_correctly_std_c']
answered_correctly_sum_c = train[['content_id','answered_correctly']].groupby(['content_id']).agg('sum').reset_index()
answered_correctly_sum_c.columns = ['content_id','answered_correctly_sum_c']
lag_time_avg_c = train[['content_id','lag_time']].groupby(['content_id']).agg('mean').reset_index()
lag_time_avg_c.columns = ['content_id','lag_time_avg_c']

answered_correctly_sum_m = train[['community','answered_correctly']].groupby(['community']).agg('sum').reset_index()
answered_correctly_sum_m.columns = ['community','answered_correctly_sum_m']
lag_time_avg_m = train[['community','lag_time']].groupby(['community']).agg('mean').reset_index()
lag_time_avg_m.columns = ['community','lag_time_avg_m']

answered_correctly_avg_p = train[['part','answered_correctly']].groupby(['part']).agg('mean').reset_index()
answered_correctly_avg_p.columns = ['part','answered_correctly_avg_p']
answered_correctly_sum_p = train[['part','answered_correctly']].groupby(['part']).agg('sum').reset_index()
answered_correctly_sum_p.columns = ['part','answered_correctly_sum_p']
lag_time_avg_p = train[['part','lag_time']].groupby(['part']).agg('mean').reset_index()
lag_time_avg_p.columns = ['part','lag_time_avg_p']

answered_correctly_avg_b = train[['bundle_id','answered_correctly']].groupby(['bundle_id']).agg('mean').reset_index()
answered_correctly_avg_b.columns = ['bundle_id','answered_correctly_avg_b']
lag_time_avg_b = train[['bundle_id','lag_time']].groupby(['bundle_id']).agg('mean').reset_index()
lag_time_avg_b.columns = ['bundle_id','lag_time_avg_b']

answered_correctly_avg_t = train[['task_container_id','answered_correctly']].groupby(['task_container_id']).agg('mean').reset_index()
answered_correctly_avg_t.columns = ['task_container_id','answered_correctly_avg_t']
answered_correctly_sum_t = train[['task_container_id','answered_correctly']].groupby(['task_container_id']).agg('sum').reset_index()
answered_correctly_sum_t.columns = ['task_container_id','answered_correctly_sum_t']
lag_time_avg_t = train[['task_container_id','lag_time']].groupby(['task_container_id']).agg('mean').reset_index()
lag_time_avg_t.columns = ['task_container_id','lag_time_avg_t']


answered_correctly_avg_g = train[['tags_lsi','answered_correctly']].groupby(['tags_lsi']).agg('mean').reset_index()
answered_correctly_avg_g.columns = ['tags_lsi','answered_correctly_avg_g']
answered_correctly_sum_g = train[['tags_lsi','answered_correctly']].groupby(['tags_lsi']).agg('sum').reset_index()
answered_correctly_sum_g.columns = ['tags_lsi','answered_correctly_sum_g']
lag_time_avg_g = train[['tags_lsi','lag_time']].groupby(['tags_lsi']).agg('mean').reset_index()
lag_time_avg_g.columns = ['tags_lsi','lag_time_avg_g']

answered_correctly_avg_c['hard_question'] = answered_correctly_avg_c['answered_correctly_avg_c'].apply(lambda x : 1 if x < 0.60 else 0)
answered_correctly_avg_c['middle_question'] = answered_correctly_avg_c['answered_correctly_avg_c'].apply(lambda x : 1 if 0.60<=x<0.83  else 0)
answered_correctly_avg_c['easy_question'] = answered_correctly_avg_c['answered_correctly_avg_c'].apply(lambda x : 1 if x >=0.83 else 0)

train = pd.merge(train, answered_correctly_avg_c, on='content_id',  how="left")
train = pd.merge(train, answered_correctly_std_c, on='content_id',  how="left")
train = pd.merge(train, answered_correctly_sum_c, on='content_id',  how="left")
train = pd.merge(train, answered_correctly_sum_m, on='community',  how="left")
train = pd.merge(train, answered_correctly_avg_p, on='part',  how="left")
train = pd.merge(train, answered_correctly_sum_p, on='part',  how="left")
train = pd.merge(train, answered_correctly_avg_b, on='bundle_id',  how="left")
train = pd.merge(train, answered_correctly_avg_t, on='task_container_id',  how="left")
train = pd.merge(train, answered_correctly_sum_t, on='task_container_id',  how="left")
train = pd.merge(train, answered_correctly_avg_g, on='tags_lsi',  how="left")
train = pd.merge(train, answered_correctly_sum_g, on='tags_lsi',  how="left")

train = pd.merge(train, lag_time_avg_c, on='content_id',  how="left")
train = pd.merge(train, lag_time_avg_m, on='community',  how="left")
train = pd.merge(train, lag_time_avg_p, on='part',  how="left")
train = pd.merge(train, lag_time_avg_b, on='bundle_id',  how="left")
train = pd.merge(train, lag_time_avg_t, on='task_container_id',  how="left")
train = pd.merge(train, lag_time_avg_g, on='tags_lsi',  how="left")
##############################

# valid = pd.merge(valid, answered_correctly_avg_c, on='content_id',  how="left")
# valid = pd.merge(valid, answered_correctly_std_c, on='content_id',  how="left")
# valid = pd.merge(valid, answered_correctly_sum_c, on='content_id',  how="left")
# valid = pd.merge(valid, answered_correctly_sum_m, on='community',  how="left")
# valid = pd.merge(valid, answered_correctly_avg_p, on='part',  how="left")
# valid = pd.merge(valid, answered_correctly_sum_p, on='part',  how="left")
# valid = pd.merge(valid, answered_correctly_avg_b, on='bundle_id',  how="left")
# valid = pd.merge(valid, answered_correctly_avg_t, on='task_container_id',  how="left")
# valid = pd.merge(valid, answered_correctly_sum_t, on='task_container_id',  how="left")
# valid = pd.merge(valid, leature_sum, on='user_id',  how="left")
# valid = pd.merge(valid, leature_mean, on='user_id',  how="left")

cum_hard = train[['user_id','hard_question']].groupby('user_id')['hard_question'].agg(['cumsum', 'cumcount'])
cum_middle = train[['user_id','middle_question']].groupby('user_id')['middle_question'].agg(['cumsum', 'cumcount'])
cum_easy = train[['user_id','easy_question']].groupby('user_id')['easy_question'].agg(['cumsum', 'cumcount'])

cum_hard['cumcount'] = cum_hard['cumcount'] + 1
cum_middle['cumcount'] = cum_middle['cumcount'] + 1
cum_easy['cumcount'] = cum_easy['cumcount'] + 1

train['hard_question_sum'] = cum_hard['cumsum']
train['middle_question_sum'] = cum_middle['cumsum']
train['easy_question_sum'] = cum_easy['cumsum']

train['hard_question_mean'] = cum_hard['cumsum'] / cum_hard['cumcount']
train['middle_question_mean'] = cum_middle['cumsum'] / cum_middle['cumcount']
train['easy_question_mean'] = cum_easy['cumsum'] / cum_easy['cumcount']

def saveFile():
    ##
#     answered_correctly_avg_c.to_csv('./answered_correctly_avg_c.csv',index=False)
#     answered_correctly_std_c.to_csv('./answered_correctly_std_c.csv',index=False)
#     answered_correctly_sum_c.to_csv('./answered_correctly_sum_c.csv',index=False)
#     answered_correctly_sum_m.to_csv('./answered_correctly_sum_m.csv',index=False)
#     answered_correctly_avg_p.to_csv('./answered_correctly_avg_p.csv',index=False)
#     answered_correctly_sum_p.to_csv('./answered_correctly_sum_p.csv',index=False)
#     answered_correctly_avg_b.to_csv('./answered_correctly_avg_b.csv',index=False)
#     answered_correctly_avg_t.to_csv('./answered_correctly_avg_t.csv',index=False)
#     answered_correctly_sum_t.to_csv('./answered_correctly_sum_t.csv',index=False)
#     leature_sum.to_csv('./leature_sum.csv',index=False)
#     leature_mean.to_csv('./leature_mean.csv',index=False)
#     hard_question_sum.to_csv('./hard_question_sum.csv',index=False)
#     middle_question_sum.to_csv('./middle_question_sum.csv',index=False)
#     easy_question_sum.to_csv('./easy_question_sum.csv',index=False)
#     hard_question_count.to_csv('./hard_question_count.csv',index=False)
#     middle_question_count.to_csv('./middle_question_count.csv',index=False)
#     easy_question_count.to_csv('./easy_question_count.csv',index=False)
#     answered_correctly_avg_g.to_csv('./answered_correctly_avg_g.csv',index=False)
#     answered_correctly_sum_g.to_csv('./answered_correctly_sum_g.csv',index=False)
#     

    lag_time_avg_c.to_csv('./lag_time_avg_c.csv',index=False)
    lag_time_avg_m.to_csv('./lag_time_avg_m.csv',index=False)
    lag_time_avg_b.to_csv('./lag_time_avg_b.csv',index=False)
    lag_time_avg_p.to_csv('./lag_time_avg_p.csv',index=False)
    lag_time_avg_t.to_csv('./lag_time_avg_t.csv',index=False)
    lag_time_avg_g.to_csv('./lag_time_avg_g.csv',index=False)

train['hmean_user_content_accuracy'] = 2 * (
    (train['user_correctness'] * train['answered_correctly_avg_c']) /
    (train['user_correctness'] + train['answered_correctly_avg_c'])
)
train['performance'] = train['answered_correctly_avg_c'] - train['user_correctness']
valid = train[-2500000:]
train = train[:-2500000]

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
        ,'tag_cor_mean','tag_cor_max','tag_cor_min',
        'lag_time_avg_c', 'lag_time_avg_m', 'lag_time_avg_p', 'lag_time_avg_b',
       'lag_time_avg_t', 'lag_time_avg_g']

TARGET = 'answered_correctly'
 
dro_cols = list(set(train.columns) - set(FEATS))
# x_tr = train[FEATS]
y_tr = train[TARGET]
x_va = valid[FEATS]
y_va = valid[TARGET]
train.drop(dro_cols, axis=1, inplace=True)
valid.drop(dro_cols, axis=1, inplace=True)

lgb_train = lgb.Dataset(train[FEATS], y_tr)
lgb_valid = lgb.Dataset(valid[FEATS], y_va)
del train, y_tr
_=gc.collect()

def LGB_bayesian(
    num_leaves,
    min_data_in_leaf,
    feature_fraction,
    n_estimators,
    ):
    
    # LightGBM expects next three parameters need to be integer. 
    num_leaves = int(num_leaves)
    # seed = int(seed)
    min_data_in_leaf = int(min_data_in_leaf)
    n_estimators = int(n_estimators)
    
    param = {
        'objective': 'binary',
        'metric': ['auc'],
        'learning_rate': 0.1,
        'num_leaves': num_leaves,
        'subsample': 0.5,
        'feature_fraction': feature_fraction,
        'max_bin': 150,
        'min_data_in_leaf': min_data_in_leaf,
        'n_estimators': n_estimators,
    }

    clf = lgb.train(param, lgb_train, valid_sets=[lgb_train, lgb_valid],
                    num_boost_round=5000,
                    verbose_eval=0, early_stopping_rounds=50)
    
    predictions = pd.DataFrame(clf.predict(x_va.iloc[:, 1:]), index=x_va.index)
    score = roc_auc_score(y_va, predictions[0])
    
    return score

bounds_LGB = {
    'num_leaves': (2**11, 2**13),
    'min_data_in_leaf': (2**12, 2**14),
    'feature_fraction': (0.2, 0.65),
    'n_estimators': (150, 5000),
    }

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=42)
init_points = 2
n_iter = 3
print('-' * 130)
import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

lgbm_params = {
    'objective': 'binary',
    'metric': ['auc'],
    'learning_rate': 0.1,
    'num_leaves': int(LGB_BO.max['params']['num_leaves']),
    'subsample': 0.5,
    'feature_fraction': LGB_BO.max['params']['feature_fraction'],
    'max_bin': 150,
    # 'n_jobs':45,
    'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
    'n_estimators': int(LGB_BO.max['params']['n_estimators']),
}
model = lgb.train(
                    lgbm_params, 
                    lgb_train,
                    valid_sets=[lgb_train, lgb_valid],
                    verbose_eval=50,
                    num_boost_round=5000,
                    early_stopping_rounds=50
                    
                    
                )
print('auc:', roc_auc_score(y_va, model.predict(valid[FEATS])))
_ = lgb.plot_importance(model)

import matplotlib.pyplot  as plt
fig,ax = plt.subplots(figsize=(15,15))
lgb.plot_importance(model,ax=ax)
plt.savefig(join(save_path, f'lgb_importance.jpg'))

import pickle 
with open('lightgbm_model.pickle','wb') as fw:
    pickle.dump(model,fw)
