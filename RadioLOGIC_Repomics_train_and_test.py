import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, RobertaTokenizer, RobertaModel, RobertaForTokenClassification, RobertaTokenizerFast
from sklearn.metrics import multilabel_confusion_matrix as mcm, classification_report  
from tqdm.auto import tqdm
import shutil, sys

df_raw =  pd.read_excel(".../data.xlsx")
df_raw_t =  pd.read_excel(".../data_test.xlsx")

df_raw['target_list1'] = df_raw[['mass', 'cyst', 'duct', 'calcification', 'fibro', 'architectural_distortion', 'skin', 'lymph_node']].values.tolist()
df_raw['target_list2'] = df_raw['us_edge'].values.tolist()
df_raw['target_list3'] = df_raw['us_shape'].values.tolist()
df_raw['target_list4'] = df_raw[['Coarse_Calcifications', 'Microcalcifications', 'Punctate', 'Amorphous', 'Pleomorphic', 'Linear', 'Branched', 'Scattered', 'Diffuse', 'Segmental', 'Clustered']].values.tolist()
df_raw['target_list5'] = df_raw['mg_density'].values.tolist()
df_raw['target_list6'] = df_raw['birads'].values.tolist()

df_raw_t['target_list1'] = df_raw_t[['mass', 'cyst', 'duct', 'calcification', 'fibro','architectural_distortion', 'skin', 'lymph_node']].values.tolist()
df_raw_t['target_list2'] = df_raw_t['us_edge'].values.tolist()
df_raw_t['target_list3'] = df_raw_t['us_shape'].values.tolist()
df_raw_t['target_list4'] = df_raw_t[['Coarse_Calcifications', 'Microcalcifications', 'Punctate', 'Amorphous', 'Pleomorphic', 'Linear', 'Branched', 'Scattered', 'Diffuse', 'Segmental', 'Clustered']].values.tolist()
df_raw_t['target_list5'] = df_raw_t['mg_density'].values.tolist()
df_raw_t['target_list6'] = df_raw_t['birads'].values.tolist()

df_train = df_raw[['verslag', 'BIO', 'target_list1', 'target_list2','target_list3','target_list4','target_list5','target_list6']].copy()
df_test = df_raw_t[['verslag', 'BIO', 'target_list1', 'target_list2','target_list3','target_list4','target_list5','target_list6']].copy()

MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizerFast.from_pretrained('.../radiobert_BigDataset_epoch10', return_tensors='pt')

labels = [i.split() for i in df_raw['BIO'].values.tolist()]   ### BIO labels
unique_labels = set()
for lb in labels:
  [unique_labels.add(i) for i in lb if i not in unique_labels]

labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

### example
text = df_raw['verslag'].values.tolist()
example = text[36]
print(example)
text_tokenized = tokenizer(example, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
word_ids = text_tokenized.word_ids()
print(tokenizer.convert_ids_to_tokens(text_tokenized["input_ids"][0]))
print(word_ids)
label_all_tokens = False


def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.title = dataframe['verslag']

        self.max_len = max_len
        lb = [i.split() for i in dataframe['label'].values.tolist()]
        txt = dataframe['verslag'].values.tolist()
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):
        return len(self.title)
        
    def get_batch_labels(self, index):
        return torch.LongTensor(self.labels[index])

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        
        #torch.tensor(self.labels[index], dtype=torch.long),

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels'  : self.get_batch_labels(index),
        }
      
class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):

        lb = [i.split() for i in df['label'].values.tolist()]
        txt = df['verslag'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]
        self.targets1 = df['target_list1']
        self.targets2 = df['target_list2']
        self.targets3 = df['target_list3']
        self.targets4 = df['target_list4']
        self.targets6 = df['target_list6']
        self.targets7 = df['target_list7']

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])


    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return {
            'batch_data': batch_data,
            'batch_labels': batch_labels, 
            'batch_targets_1': torch.tensor(self.targets1[idx], dtype=torch.float), 
            'batch_targets_2': torch.tensor(self.targets2[idx], dtype=torch.long),
            'batch_targets_3': torch.tensor(self.targets3[idx], dtype=torch.long),
            'batch_targets_4': torch.tensor(self.targets4[idx], dtype=torch.float),
            'batch_targets_5': torch.tensor(self.targets5[idx], dtype=torch.long),
            'batch_targets_6': torch.tensor(self.targets6[idx], dtype=torch.long),
        }

  
train_dataset = df_train.sample(frac=1,random_state=1203)
valid_dataset = df_test.sample(frac=1,random_state=1203)
train_dataset = DataSequence(df_train)
val_dataset = DataSequence(df_test)

train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=8, batch_size=16)
print("train_dataloader:",len(train_dataloader))

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

class RadioLOGIC(torch.nn.Module):

    def __init__(self):

        super(Radio_RoBERTa, self).__init__()

        self.bert = RobertaModel.from_pretrained('.../RadioLOGIC_15p', add_pooling_layer=False) #/home/t.zhang/NLP/radiobert_BigDataset_epoch10
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 39)
        self.num_labels=39
        
        self.l21 = torch.nn.Dropout(0.3)
        self.l31 = torch.nn.Linear(768, 8)
        self.l32 = torch.nn.Linear(768, 3)
        self.l33 = torch.nn.Linear(768, 3)
        self.l34 = torch.nn.Linear(768, 11)
        self.l35 = torch.nn.Linear(768, 5)
        self.l36 = torch.nn.Linear(768, 6)


    def forward(self, input_id=None, attention_mask=None, labels=None): #, return_dict=None

        outputs = self.bert(input_ids=input_id, attention_mask=attention_mask, return_dict=False)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        pooler = sequence_output[:, 0]
        output_21 = self.l21(pooler)
        output1 = self.l31(output_21)
        output2 = self.l32(output_21)
        output3 = self.l33(output_21)
        output4 = self.l34(output_21)
        output5 = self.l35(output_21)
        output6 = self.l36(output_21)        
        
        
        loss = None
        if labels is not None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            nSamples = [number_1, ... , number_n]  ### weights based on numbers 
            normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
            weights = torch.FloatTensor(normedWeights).to(device, dtype = torch.float)

            loss_fct = torch.nn.CrossEntropyLoss() #weight=weights
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #if not return_dict:
            #output = (logits,) + outputs[2:]
            #return ((loss,) + output) if loss is not None else output
        output = (logits,) + outputs[2:]  

        return ((loss,) + output + (output1,)+ (output2,)+ (output3,)+ (output4,)+ (output5,)+(output6,)) if loss is not None else (output + (output1,)+ (output2,)+ (output3,)+ (output4,)+ (output6,)+(output7,))

def loss_fn1(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def loss_fn2(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def train_loop(model, train_dataloader, val_dataloader, checkpoint_path, best_model_path):

    optimizer = torch.optim.SGD(params =  model.parameters(), lr=LEARNING_RATE)
    valid_acc_min = 0
    valid_f1_min = 0
        
    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0
        train_targets1=[]
        train_outputs1=[]
        train_targets2=[]
        train_outputs2=[] 
        train_targets3=[]
        train_outputs3=[] 
        train_targets4=[]
        train_outputs4=[] 
        train_targets5=[]
        train_outputs5=[] 
        train_targets6=[]
        train_outputs6=[]        

        model.train()
        print('############# Epoch {}: Training Start   #############'.format(epoch_num+1))
        loop = tqdm(train_dataloader,leave=True)

        for batch in loop:

            train_label = batch['batch_labels'].to(device)
            mask = batch['batch_data']['attention_mask'][:,0,:].to(device)
            input_id = batch['batch_data']['input_ids'][:,0,:].to(device)
            targets1 = batch['batch_targets_1'].to(device, dtype = torch.float)
            targets2 = batch['batch_targets_2'].to(device, dtype = torch.long)
            targets3 = batch['batch_targets_3'].to(device, dtype = torch.long)
            targets4 = batch['batch_targets_4'].to(device, dtype = torch.float)
            targets5 = batch['batch_targets_5'].to(device, dtype = torch.long)
            targets6 = batch['batch_targets_6'].to(device, dtype = torch.long)            

            optimizer.zero_grad()
            loss_l, logits,output1, output2, output3, output4, output5, output6= model(input_id, mask, train_label)
            
            nSamples1 = [number_1, ... , number_n]  ### weights based on numbers, n=8 here.
            normedWeights1 = [1 - (x / sum(nSamples1)) for x in nSamples1]
            weights1 = torch.FloatTensor(normedWeights1).to(device, dtype = torch.float)
            loss_fct1 = torch.nn.BCEWithLogitsLoss(weight=weights1)
            loss1 = loss_fct1(output1, targets1)
            
            nSamples2 = [number_1, ... , number_n]  ### weights based on numbers, n=3 here. 
            normedWeights2 = [1 - (x / sum(nSamples2)) for x in nSamples2]
            weights2 = torch.FloatTensor(normedWeights2).to(device, dtype = torch.float)
            loss_fct2 = torch.nn.CrossEntropyLoss(weight=weights2)            
            loss2 = loss_fct2(output2, targets2)
            
            nSamples3 = [number_1, ... , number_n]  ### weights based on numbers, n=3 here.
            normedWeights3 = [1 - (x / sum(nSamples3)) for x in nSamples3]
            weights3 = torch.FloatTensor(normedWeights3).to(device, dtype = torch.float)
            loss_fct3 = torch.nn.CrossEntropyLoss(weight=weights3)    
            loss3 = loss_fct3(output3, targets3)
            
            nSamples4 = [number_1, ... , number_n]  ### weights based on numbers, n=11 here. 
            normedWeights4 = [1 - (x / sum(nSamples4)) for x in nSamples4]
            weights4 = torch.FloatTensor(normedWeights4).to(device, dtype = torch.float)
            loss_fct4 = torch.nn.BCEWithLogitsLoss(weight=weights4)
            loss4 = loss_fct4(output4, targets4)

            
            nSamples5 = [number_1, ... , number_n]  ### weights based on numbers, n=5 here.
            normedWeights5 = [1 - (x / sum(nSamples5)) for x in nSamples5]
            weights5 = torch.FloatTensor(normedWeights5).to(device, dtype = torch.float)
            loss_fct5 = torch.nn.CrossEntropyLoss(weight=weights5)   
            loss5 = loss_fct5(output5, targets5)
            
            nSamples6 = [number_1, ... , number_n]  ### weights based on numbers, n=6 here.
            normedWeights6 = [1 - (x / sum(nSamples6)) for x in nSamples6]
            weights6 = torch.FloatTensor(normedWeights6).to(device, dtype = torch.float)
            loss_fct6 = torch.nn.CrossEntropyLoss(weight=weights6)
            loss6 = loss_fct6(output6, targets6)
            
            loss=loss_l + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            train_targets1.extend(targets1.cpu().detach().numpy().tolist())
            train_outputs1.extend(torch.sigmoid(output1).cpu().detach().numpy().tolist())
            
            train_targets2.extend(targets2.cpu().detach().numpy().tolist())
            _, predicted2 = torch.max(output2.data, dim=1)
            train_outputs2.extend(predicted2.cpu().detach().numpy().tolist())
            
            train_targets3.extend(targets3.cpu().detach().numpy().tolist())
            _, predicted3 = torch.max(output3.data, dim=1)
            train_outputs3.extend(predicted3.cpu().detach().numpy().tolist())
            
            train_targets4.extend(targets4.cpu().detach().numpy().tolist())
            train_outputs4.extend(torch.sigmoid(output4).cpu().detach().numpy().tolist())      
            
            train_targets5.extend(targets5.cpu().detach().numpy().tolist())
            _, predicted5 = torch.max(output5.data, dim=1)
            train_outputs5.extend(predicted5.cpu().detach().numpy().tolist())   
            
            train_targets6.extend(targets6.cpu().detach().numpy().tolist())
            _, predicted6 = torch.max(output6.data, dim=1)
            train_outputs6.extend(predicted6.cpu().detach().numpy().tolist())                   

            logits_clean = logits[train_label != -100]
            label_clean = train_label[train_label != -100]

            predictions = logits_clean.argmax(dim=1)
            acc = metrics.accuracy_score(label_clean.cpu().detach().numpy().tolist(), predictions.cpu().detach().numpy().tolist())
            total_acc_train += acc
            total_loss_train += loss.item()


            loss.backward()
            optimizer.step()

        train_preds1 = (np.array(train_outputs1) > 0.5).astype(int)
        train_preds4 = (np.array(train_outputs4) > 0.5).astype(int)    
        f1_train_micro_1 = metrics.f1_score(train_targets1, train_preds1, average='micro')
        f1_train_micro_2 = metrics.f1_score(train_targets2, train_outputs2, average='micro')
        f1_train_micro_3 = metrics.f1_score(train_targets3, train_outputs3, average='micro')
        f1_train_micro_4 = metrics.f1_score(train_targets4, train_preds4, average='micro')
        f1_train_micro_5 = metrics.f1_score(train_targets5, train_outputs5, average='micro')
        f1_train_micro_6 = metrics.f1_score(train_targets6, train_outputs6, average='micro')
        
        f1_train_weighted_1 = metrics.f1_score(train_targets1, train_preds1, average='weighted')
        f1_train_weighted_2 = metrics.f1_score(train_targets2, train_outputs2, average='weighted')
        f1_train_weighted_3 = metrics.f1_score(train_targets3, train_outputs3, average='weighted')
        f1_train_weighted_4 = metrics.f1_score(train_targets4, train_preds4, average='weighted')
        f1_train_weighted_5 = metrics.f1_score(train_targets5, train_outputs5, average='weighted')
        f1_train_weighted_6 = metrics.f1_score(train_targets6, train_outputs6, average='weighted')        
        train_f1=(f1_train_micro_1+f1_train_micro_2+f1_train_micro_3+f1_train_micro_4+f1_train_micro_5+f1_train_micro_6)/6
        train_f1_w=(f1_train_weighted_1+f1_train_weighted_2+f1_train_weighted_3+f1_train_weighted_4+f1_train_weighted_5+f1_train_weighted_6)/6
        print(
            f'Train_loss: | loss_l: {loss_l: .3f} | loss1: {loss1: .3f} | loss2: {loss2: .3f} | loss3: {loss3: .3f} | loss4: {loss4: .3f} | loss5: {loss5: .3f} | loss6: {loss6: .3f}')    
        model.eval()
        print('############# Epoch {}: Validation Start   #############'.format(epoch_num+1))

        total_acc_val = 0
        total_loss_val = 0
        val_targets1=[]
        val_outputs1=[]
        val_targets2=[]
        val_outputs2=[] 
        val_targets3=[]
        val_outputs3=[] 
        val_targets4=[]
        val_outputs4=[] 
        val_targets5=[]
        val_outputs5=[] 
        val_targets6=[]
        val_outputs6=[] 

        loop_v=tqdm(val_dataloader,leave=True)

        for batch in loop_v:

            val_label = batch['batch_labels'].to(device)
            mask = batch['batch_data']['attention_mask'][:,0,:].to(device)
            input_id = batch['batch_data']['input_ids'][:,0,:].to(device)
            targets1 = batch['batch_targets_1'].to(device, dtype = torch.float)
            targets2 = batch['batch_targets_2'].to(device, dtype = torch.long)
            targets3 = batch['batch_targets_3'].to(device, dtype = torch.long)
            targets4 = batch['batch_targets_4'].to(device, dtype = torch.float)
            targets5 = batch['batch_targets_5'].to(device, dtype = torch.long)
            targets6 = batch['batch_targets_6'].to(device, dtype = torch.long)              
            loss, logits,output1, output2, output3, output4, output5, output6 = model(input_id, mask, val_label)
            loss1 = loss_fn1(output1, targets1)
            loss2 = loss_fn2(output2, targets2)
            loss3 = loss_fn2(output3, targets3)
            loss4 = loss_fn1(output4, targets4)
            loss5 = loss_fn2(output5, targets5)
            loss6 = loss_fn2(output6, targets6)                   
            loss=loss_l + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            val_targets1.extend(targets1.cpu().detach().numpy().tolist())
            val_outputs1.extend(torch.sigmoid(output1).cpu().detach().numpy().tolist())

            val_targets2.extend(targets2.cpu().detach().numpy().tolist())
            _, predicted2 = torch.max(output2.data, dim=1)
            val_outputs2.extend(predicted2.cpu().detach().numpy().tolist())

            val_targets3.extend(targets3.cpu().detach().numpy().tolist())
            _, predicted3 = torch.max(output3.data, dim=1)
            val_outputs3.extend(predicted3.cpu().detach().numpy().tolist())

            val_targets4.extend(targets4.cpu().detach().numpy().tolist())
            val_outputs4.extend(torch.sigmoid(output4).cpu().detach().numpy().tolist())            

            val_targets5.extend(targets5.cpu().detach().numpy().tolist())
            _, predicted5 = torch.max(output5.data, dim=1)
            val_outputs6.extend(predicted5.cpu().detach().numpy().tolist())   

            val_targets6.extend(targets6.cpu().detach().numpy().tolist())
            _, predicted6 = torch.max(output6.data, dim=1)
            val_outputs6.extend(predicted6.cpu().detach().numpy().tolist())
            
            logits_clean = logits[val_label != -100]
            label_clean = val_label[val_label != -100]
            predictions = logits_clean.argmax(dim=1)          
            acc = metrics.accuracy_score(label_clean.cpu().detach().numpy().tolist(), predictions.cpu().detach().numpy().tolist())
            total_acc_val += acc
            total_loss_val += loss.item()

        print(
            f'Valid_loss: | loss_l: {loss_l: .3f} | loss1: {loss1: .3f} | loss2: {loss2: .3f} | loss3: {loss3: .3f} | loss4: {loss4: .3f} | loss5: {loss5: .3f} | loss6: {loss6: .3f}')  
                  
        val_preds1 = (np.array(val_outputs1) > 0.5).astype(int)
        val_preds4 = (np.array(val_outputs4) > 0.5).astype(int)
        f1_valid_micro_1 = metrics.f1_score(val_targets1, val_preds1, average='micro')
        f1_valid_micro_2 = metrics.f1_score(val_targets2, val_outputs2, average='micro')
        f1_valid_micro_3 = metrics.f1_score(val_targets3, val_outputs3, average='micro')
        f1_valid_micro_4 = metrics.f1_score(val_targets4, val_preds4, average='micro')
        f1_valid_micro_5 = metrics.f1_score(val_targets5, val_outputs5, average='micro')
        f1_valid_micro_6 = metrics.f1_score(val_targets6, val_outputs6, average='micro')
        
        f1_valid_weighted_1 = metrics.f1_score(val_targets1, val_preds1, average='weighted')
        f1_valid_weighted_2 = metrics.f1_score(val_targets2, val_outputs2, average='weighted')
        f1_valid_weighted_3 = metrics.f1_score(val_targets3, val_outputs3, average='weighted')
        f1_valid_weighted_4 = metrics.f1_score(val_targets4, val_preds4, average='weighted')
        f1_valid_weighted_5 = metrics.f1_score(val_targets5, val_outputs5, average='weighted')
        f1_valid_weighted_6 = metrics.f1_score(val_targets6, val_outputs6, average='weighted')   
                
        valid_f1=(f1_valid_micro_1+f1_valid_micro_2+f1_valid_micro_3+f1_valid_micro_4+f1_valid_micro_5+f1_valid_micro_6)/6          
        valid_f1_w=(f1_valid_weighted_1+f1_valid_weighted_2+f1_valid_weighted_3+f1_valid_weighted_4+f1_valid_weighted_5+f1_valid_weighted_6)/6  
             
        train_loss=total_loss_train / len(loop)
        valid_loss=total_loss_val / len(loop_v)
        train_acc =total_acc_train / len(loop)
        valid_acc =total_acc_val / len(loop_v)
        
        print(
            f'Epochs: {epoch_num + 1} | Train_Loss: {total_loss_train / len(loop): .3f} | Train_Accuracy: {total_acc_train / len(loop): .3f} | Train_F1: {train_f1: .3f} | Train_F1_w: {train_f1_w: .3f} | Val_Loss: {total_loss_val / len(loop_v): .3f} | Val_Accuracy: {total_acc_val / len(loop_v): .3f} | Val_F1: {valid_f1: .3f} | Val_F1_w: {valid_f1_w: .3f}')
        checkpoint = {
            'epoch': epoch_num + 1,
            'valid_loss_min': valid_loss,
            'valid_acc_max': valid_acc,
            'valid_f1_max': valid_f1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}
        valid_acc =total_acc_val / len(loop_v)
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        if valid_acc_min <= valid_acc:
            print('Validation acc increased ({:.6f} --> {:.6f}).  Want to save model? ...'.format(valid_acc_min,valid_acc))
            #save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_acc_min = valid_acc
            
        if valid_f1_min <= valid_f1:
            print('Validation f1 increased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_f1_min,valid_f1))
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_f1_min = valid_f1
        print('---------------------------------------------------------------------------------------------------')  
 
LEARNING_RATE = 1e-02
EPOCHS = 200
model = RadioLOGIC()
model.to(device)
print(model)

checkpoint_path = '.../checkpoint/current_checkpoint.pt'
best_model = '.../best_model/best_model.pt'

trained_model = train_loop(model, train_dataloader, val_dataloader,checkpoint_path, best_model)






### test!!!
print('---------------------------------------------------------------------------------------------------')
print("Testing...")
checkpoint = torch.load(".../best_model/best_model.pt")
try:
    checkpoint.eval()
except AttributeError as error:
    print("error")

model.load_state_dict(checkpoint['state_dict'])
model.eval()

def align_word_ids(texts):
  
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

def evaluate_one_text(model, sentence):
  
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'][0].unsqueeze(0).to(device)

    input_id = text['input_ids'][0].unsqueeze(0).to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    print(sentence)
    print(prediction_label)
    
a='...' ### radiological reports
evaluate_one_text(model, a) ### example to evaluate BIO labels



print("test: -----------------------------------------------------------------------------------------------------")
val_targets1=[]
val_outputs1=[]
val_targets2=[]
val_outputs2=[] 
val_targets3=[]
val_outputs3=[] 
val_targets4=[]
val_outputs4=[] 
val_targets5=[]
val_outputs5=[] 
val_targets6=[]
val_outputs6=[]
model = RadioLOGIC()
model.to(device)
checkpoint = torch.load(".../best_model/best_model.pt")
try:
    checkpoint.eval()
except AttributeError as error:
    print("error")

model.load_state_dict(checkpoint['state_dict'])
### now you can evaluate it
model.eval()

with torch.no_grad():
      #for batch_idx, data in enumerate(validation_loader, 0):
      loop_v=tqdm(val_dataloader,leave=True)
      for batch in loop_v:
      
            val_label = batch['batch_labels'].to(device)
            mask = batch['batch_data']['attention_mask'][:,0,:].to(device)
            input_id = batch['batch_data']['input_ids'][:,0,:].to(device)
            targets1 = batch['batch_targets_1'].to(device, dtype = torch.float)
            targets2 = batch['batch_targets_2'].to(device, dtype = torch.long)
            targets3 = batch['batch_targets_3'].to(device, dtype = torch.long)
            targets4 = batch['batch_targets_4'].to(device, dtype = torch.float)
            targets5 = batch['batch_targets_5'].to(device, dtype = torch.long)
            targets6 = batch['batch_targets_6'].to(device, dtype = torch.long)              
            loss, logits,output1, output2, output3, output4, output5, output6 = model(input_id, mask, val_label)

            val_targets1.extend(targets1.cpu().detach().numpy().tolist())
            val_outputs1.extend(torch.sigmoid(output1).cpu().detach().numpy().tolist())

            val_targets2.extend(targets2.cpu().detach().numpy().tolist())
            _, predicted2 = torch.max(output2.data, dim=1)
            val_outputs2.extend(predicted2.cpu().detach().numpy().tolist())

            val_targets3.extend(targets3.cpu().detach().numpy().tolist())
            _, predicted3 = torch.max(output3.data, dim=1)
            val_outputs3.extend(predicted3.cpu().detach().numpy().tolist())

            val_targets4.extend(targets4.cpu().detach().numpy().tolist())
            val_outputs4.extend(torch.sigmoid(output4).cpu().detach().numpy().tolist())           

            val_targets5.extend(targets5.cpu().detach().numpy().tolist())
            _, predicted5 = torch.max(output5.data, dim=1)
            val_outputs5.extend(predicted5.cpu().detach().numpy().tolist())   

            val_targets6.extend(targets6.cpu().detach().numpy().tolist())
            _, predicted6 = torch.max(output6.data, dim=1)
            val_outputs6.extend(predicted6.cpu().detach().numpy().tolist())

val_preds1 = (np.array(val_outputs1) > 0.5).astype(int)
val_outputs4 = (np.array(val_outputs4) > 0.5).astype(int)  

labels1 =['mass', 'cyst', 'duct', 'calcification', 'fibro', 'architectural_distortion', 'skin', 'lymph_node']
print(classification_report(val_targets1, val_preds1, target_names=labels1))
accuracy_1 = metrics.accuracy_score(val_targets1, val_preds1)
f1_score_micro_1 = metrics.f1_score(val_targets1, val_preds1, average='micro')
f1_score_macro_1 = metrics.f1_score(val_targets1, val_preds1, average='macro')
f1_score_weighted_1 = metrics.f1_score(val_targets1, val_preds1, average='weighted')
print(f"Accuracy Score_1 = {accuracy_1}")
print(f"F1 Score (Micro)_1 = {f1_score_micro_1}")
print(f"F1 Score (Macro)_1 = {f1_score_macro_1}")
print(f"F1 Score (Weighted)_1 = {f1_score_weighted_1}")
print("----------------------------------------------------------------------------------")

labels2 =['NA', 'irregular', 'circumscribed']
print(classification_report(val_targets2, val_outputs2, target_names=labels2))
accuracy_2 = metrics.accuracy_score(val_targets2, val_outputs2)
f1_score_micro_2 = metrics.f1_score(val_targets2, val_outputs2, average='micro')
f1_score_macro_2 = metrics.f1_score(val_targets2, val_outputs2, average='macro')
f1_score_weighted_2 = metrics.f1_score(val_targets2, val_outputs2, average='weighted')
print(f"Accuracy Score_2 = {accuracy_2}")
print(f"F1 Score (Micro)_2 = {f1_score_micro_2}")
print(f"F1 Score (Macro)_2 = {f1_score_macro_2}")
print(f"F1 Score (Weighted)_2 = {f1_score_weighted_2}")
print("----------------------------------------------------------------------------------")

labels3 =['NA', 'irregular', 'oval/round']
print(classification_report(val_targets3, val_outputs3, target_names=labels3))
accuracy_3 = metrics.accuracy_score(val_targets3, val_outputs3)
f1_score_micro_3 = metrics.f1_score(val_targets3, val_outputs3, average='micro')
f1_score_macro_3 = metrics.f1_score(val_targets3, val_outputs3, average='macro')
f1_score_weighted_3 = metrics.f1_score(val_targets3, val_outputs3, average='weighted')
print(f"Accuracy Score_3 = {accuracy_3}")
print(f"F1 Score (Micro)_3 = {f1_score_micro_3}")
print(f"F1 Score (Macro)_3 = {f1_score_macro_3}")
print(f"F1 Score (Weighted)_3 = {f1_score_weighted_3}")
print("----------------------------------------------------------------------------------")

labels4 =['Coarse_Calcifications', 'Microcalcifications',	'Punctate',	'Amorphous',	'Pleomorphic',	'Linear',	 'Branched', 	'Scattered',	'Diffuse', 	'Segmental',	'Clustered']
print(classification_report(val_targets4, val_outputs4, target_names=labels4))
accuracy_4 = metrics.accuracy_score(val_targets4, val_outputs4)
f1_score_micro_4 = metrics.f1_score(val_targets4, val_outputs4, average='micro')
f1_score_macro_4 = metrics.f1_score(val_targets4, val_outputs4, average='macro')
f1_score_weighted_4 = metrics.f1_score(val_targets4, val_outputs4, average='weighted')
print(f"Accuracy Score_4 = {accuracy_4}")
print(f"F1 Score (Micro)_4 = {f1_score_micro_4}")
print(f"F1 Score (Macro)_4 = {f1_score_macro_4}")
print(f"F1 Score (Weighted)_4 = {f1_score_weighted_4}")
print("----------------------------------------------------------------------------------")

labels5 =['NA', 'ACR 1', 'ACR 2', 'ACR 3', 'ACR 4']
print(classification_report(val_targets5, val_outputs5, target_names=labels5))
accuracy_5 = metrics.accuracy_score(val_targets5, val_outputs5)
f1_score_micro_5 = metrics.f1_score(val_targets5, val_outputs5, average='micro')
f1_score_macro_5 = metrics.f1_score(val_targets5, val_outputs5, average='macro')
f1_score_weighted_5 = metrics.f1_score(val_targets5, val_outputs5, average='weighted')
print(f"Accuracy Score_5 = {accuracy_5}")
print(f"F1 Score (Micro)_5 = {f1_score_micro_5}")
print(f"F1 Score (Macro)_5 = {f1_score_macro_5}")
print(f"F1 Score (Weighted)_5 = {f1_score_weighted_5}")

labels6 =['no', 'BIRADS 1', 'BIRADS 2', 'BIRADS 3', 'BIRADS 4', 'BIRADS 5']
print(classification_report(val_targets6, val_outputs6, target_names=labels6))
accuracy_6 = metrics.accuracy_score(val_targets6, val_outputs6)
f1_score_micro_6 = metrics.f1_score(val_targets6, val_outputs6, average='micro')
f1_score_macro_6 = metrics.f1_score(val_targets6, val_outputs6, average='macro')
f1_score_weighted_6 = metrics.f1_score(val_targets6, val_outputs6, average='weighted')
print(f"Accuracy Score_6 = {accuracy_6}")
print(f"F1 Score (Micro)_6 = {f1_score_micro_6}")
print(f"F1 Score (Macro)_6 = {f1_score_macro_6}")
print(f"F1 Score (Weighted)_6 = {f1_score_weighted_6}")
print("----------------------------------------------------------------------------------")
