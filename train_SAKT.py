import gc
import psutil
import joblib
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# %% [code]
TRAIN_SAMPLES = 320000
MAX_SEQ = 240
MIN_SAMPLES = 5
EMBED_DIM = 256
DROPOUT_RATE = 0.1
LEARNING_RATE = 1e-3
MAX_LEARNING_RATE = 2e-3
EPOCHS = 30
TRAIN_BATCH_SIZE = 2048


dtypes = {'timestamp': 'int64', 'user_id': 'int32' ,'content_id': 'int16','content_type_id': 'int8','answered_correctly':'int8'}
train_df = pd.read_feather('../input/riiid-cross-validation-dataset/train.feather')[[
    'timestamp', 'user_id', 'content_id', 'content_type_id', 'answered_correctly'
]]
for col, dtype in dtypes.items():
    train_df[col] = train_df[col].astype(dtype)
train_df = train_df[train_df.content_type_id == False]
train_df = train_df.sort_values(['timestamp'], ascending=True)
train_df.reset_index(drop=True, inplace=True)

skills = train_df["content_id"].unique()
joblib.dump(skills, "skills.pkl.zip")
n_skill = len(skills)
print("number skills", len(skills))

group = train_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values))

joblib.dump(group, "group.pkl.zip")
del train_df
gc.collect()

train_indexes = list(group.index)[:TRAIN_SAMPLES]
valid_indexes = list(group.index)[TRAIN_SAMPLES:]
train_group = group[group.index.isin(train_indexes)]
valid_group = group[group.index.isin(valid_indexes)]
del group, train_indexes, valid_indexes
print(len(train_group), len(valid_group))

class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, min_samples=1, max_seq=128):
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = {}
        
        self.user_ids = []
        for user_id in group.index:
            q, qa = group[user_id]
            if len(q) < min_samples:
                continue
            
            # Main Contribution
            if len(q) > self.max_seq:
                total_questions = len(q)
                initial = total_questions % self.max_seq
                if initial >= min_samples:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (q[:initial], qa[:initial])
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = start + self.max_seq
                    self.samples[f"{user_id}_{seq+1}"] = (q[start:end], qa[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (q, qa)
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_ = self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        if seq_len == self.max_seq:
            q[:] = q_
            qa[:] = qa_
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_
        
        target_id = q[1:]
        label = qa[1:]

        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill

        return x, target_id, label

train_dataset = SAKTDataset(train_group, n_skill, min_samples=MIN_SAMPLES, max_seq=MAX_SEQ)
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)
valid_dataset = SAKTDataset(valid_group, n_skill, max_seq=MAX_SEQ)
valid_dataloader = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)

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

def train_fn(model, dataloader, optimizer, scheduler, criterion, device="cpu"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    for item in dataloader:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()
        label = item[2].to(device).float()
        target_mask = (target_id != 0)

        optimizer.zero_grad()
        output, _, = model(x, target_id)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss.append(loss.item())

        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)
        pred = (torch.sigmoid(output) >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc

def valid_fn(model, dataloader, criterion, device="cpu"):
    model.eval()

    valid_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    for item in dataloader:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()
        label = item[2].to(device).float()
        target_mask = (target_id != 0)

        output, _, = model(x, target_id)
        loss = criterion(output, label)
        valid_loss.append(loss.item())

        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)
        pred = (torch.sigmoid(output) >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(valid_loss)

    return loss, acc, auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAKTModel(n_skill, max_seq=MAX_SEQ, embed_dim=EMBED_DIM, dropout_rate=DROPOUT_RATE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
)

model.to(device)
criterion.to(device)

best_auc = 0
max_steps = 3
step = 0
for epoch in range(EPOCHS):
    loss, acc, auc = train_fn(model, train_dataloader, optimizer, scheduler, criterion, device)
    print("epoch - {}/{} train: - {:.3f} acc - {:.3f} auc - {:.3f}".format(epoch+1, EPOCHS, loss, acc, auc))
    loss, acc, auc = valid_fn(model, valid_dataloader, criterion, device)
    print("epoch - {}/{} valid: - {:.3f} acc - {:.3f} auc - {:.3f}".format(epoch+1, EPOCHS, loss, acc, auc))
    if auc > best_auc:
        best_auc = auc
        step = 0
        torch.save(model.state_dict(), "sakt_model.pt")
    else:
        step += 1
        if step >= max_steps:
            break

del train_dataset, valid_dataset

torch.save(model.state_dict(), "sakt_model_final.pt")

# Infer the test dataset

# class TestDataset(Dataset):
#     def __init__(self, samples, test_df, skills, max_seq=MAX_SEQ):
#         super(TestDataset, self).__init__()
#         self.samples = samples
#         self.user_ids = [x for x in test_df["user_id"].unique()]
#         self.test_df = test_df
#         self.skills = skills
#         self.n_skill = len(skills)
#         self.max_seq = max_seq

#     def __len__(self):
#         return self.test_df.shape[0]

#     def __getitem__(self, index):
#         test_info = self.test_df.iloc[index]

#         user_id = test_info["user_id"]
#         target_id = test_info["content_id"]

#         q = np.zeros(self.max_seq, dtype=int)
#         qa = np.zeros(self.max_seq, dtype=int)

#         if user_id in self.samples.index:
#             q_, qa_ = self.samples[user_id]
            
#             seq_len = len(q_)

#             if seq_len >= self.max_seq:
#                 q = q_[-self.max_seq:]
#                 qa = qa_[-self.max_seq:]
#             else:
#                 q[-seq_len:] = q_
#                 qa[-seq_len:] = qa_          
        
#         x = np.zeros(self.max_seq-1, dtype=int)
#         x = q[1:].copy()
#         x += (qa[1:] == 1) * self.n_skill
        
#         questions = np.append(q[2:], [target_id])
        
#         return x, questions

# import riiideducation

# env = riiideducation.make_env()
# iter_test = env.iter_test()

# group = joblib.load("group.pkl.zip")

# model.eval()
# prev_test_df = None

# for (test_df, sample_prediction_df) in tqdm(iter_test):
#     if (prev_test_df is not None) & (psutil.virtual_memory().percent < 90):
#         prev_test_df['answered_correctly'] = eval(test_df['prior_group_answers_correct'].iloc[0])
#         prev_test_df = prev_test_df[prev_test_df.content_type_id == False]
        
#         prev_group = prev_test_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
#             r['content_id'].values,
#             r['answered_correctly'].values))
#         for prev_user_id in prev_group.index:
#             if prev_user_id in group.index:
#                 group[prev_user_id] = (
#                     np.append(group[prev_user_id][0], prev_group[prev_user_id][0])[-MAX_SEQ:], 
#                     np.append(group[prev_user_id][1], prev_group[prev_user_id][1])[-MAX_SEQ:]
#                 )
 
#             else:
#                 group[prev_user_id] = (
#                     prev_group[prev_user_id][0], 
#                     prev_group[prev_user_id][1]
#                 )

#     prev_test_df = test_df.copy()
    
#     test_df = test_df[test_df.content_type_id == False]
#     test_dataset = TestDataset(group, test_df, skills)
#     test_dataloader = DataLoader(test_dataset, batch_size=51200, shuffle=False)
    
#     outs = []

#     for item in tqdm(test_dataloader):
#         x = item[0].to(device).long()
#         target_id = item[1].to(device).long()

#         with torch.no_grad():
#             output, att_weight = model(x, target_id)
#         outs.extend(torch.sigmoid(output)[:, -1].view(-1).data.cpu().numpy())
        
#     test_df['answered_correctly'] = outs
#     env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

# att_weight = att_weight.detach().cpu().numpy()
