from model import BertBidaf
from data import DataGenerator,load_data
import torch
from torch import nn, optim
from transformers import *
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_batch_size=4
predict_batch_size=4
learning_rate=2e-5
epoch=2
resume_train=True
resume_predict=True

#pretrained = 'albert_chinese_tiny'
pretrained = 'albert_chinese_xlarge'
tokenizer = BertTokenizer.from_pretrained(pretrained)


def get_batch_predict_answer(batch_token_ids, p1, p2,max_a_len=16):
    
    batch_size, c_len = p1.size()
    ls = nn.LogSoftmax(dim=1)
    mask_pos = (torch.ones(c_len, c_len).to(device) * float('-inf')).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
    mask_len = (torch.ones(c_len, c_len).to(device) * float('-inf')).triu(max_a_len).unsqueeze(0).expand(batch_size, -1, -1)

    score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask_pos + mask_len
    score, s_idx = score.max(dim=1)
    score, e_idx = score.max(dim=1)
    s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()
    answers=[]
    for i in range(batch_size):
        answer = tokenizer.decode(batch_token_ids[i][s_idx[i]:e_idx[i]+1]).replace(" ","")
        answers.append(answer)
    return answers


def train(infile):
    if resume_train==False:
        encode_model = AlbertModel.from_pretrained(pretrained).to(device)
        model=BertBidaf(tokenizer,encode_model,device)
    else:
        model=torch.load("dureder_model")
    model=model.to(device)
    data_gen=DataGenerator(infile,tokenizer,device)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for e in range(epoch):
        batch_cnt=0
        for batch in data_gen.batchIter(train_batch_size):
            batch_cnt+=1

            batch_pair_ids=batch["batch_pair_ids"]
            batch_token_type_ids=batch["batch_token_type_ids"]
            batch_attention_mask=batch["batch_attention_mask"]
            batch_start=batch["batch_start"]
            batch_end=batch["batch_end"]
            batch_context_len=batch["batch_context_len"]
            batch_question_len=batch["batch_question_len"]


            p1,p2=model(batch_pair_ids,batch_token_type_ids,batch_attention_mask,batch_context_len,batch_question_len)
            optimizer.zero_grad()
            batch_loss = criterion(p1, batch_start) + criterion(p2, batch_end)
            print(e,batch_cnt*train_batch_size,batch_loss.item())
            # print(get_batch_predict_answer(batch_pair_ids, p1, p2))
            # print(batch_ans)
            batch_loss.backward()
            optimizer.step()

            if(batch_cnt % 20 == 0):
                torch.save(model,'dureder_model')

        torch.save(model,'dureder_model')

def predict_to_file(infile, out_file="result.json"):

    def save_json(record_dict):
        fw = open(out_file, 'w', encoding='utf-8')
        json_str = json.dumps(record_dict, ensure_ascii=False, indent=4)
        fw.write(json_str)
        fw.close()

    model=torch.load("dureder_model").to(device)
    model.eval()
    
    R = {}

    ignore_id=set()
    if resume_predict:
        with open(out_file,'r') as load_f:
            R = json.load(load_f)
        for key in R:
            ignore_id.add(key)


    data_gen = DataGenerator(infile, tokenizer,device)
    batch_cnt=0
    for batch in data_gen.batchIterNoAnswer(predict_batch_size,ignore_id):
        batch_cnt+=1

        batch_id=batch["batch_id"]
        batch_pair_ids = batch["batch_pair_ids"]
        batch_token_type_ids = batch["batch_token_type_ids"]
        batch_attention_mask = batch["batch_attention_mask"]
        batch_context_len=batch["batch_context_len"]
        batch_question_len=batch["batch_question_len"]
        batch_context_ids=batch["batch_context_ids"]

        p1, p2 = model(batch_pair_ids,batch_token_type_ids,batch_attention_mask,batch_context_len,batch_question_len)
        answers=get_batch_predict_answer(batch_context_ids, p1, p2)
        print(answers)
        for i in range(len(batch_id)):
            R[batch_id[i]] = answers[i]
        if(batch_cnt % 200 == 0):
            save_json(R)
    
    save_json(R)

    


#train("../data/train.json")
#train("../data/demo/demo_train.json")


#predict_to_file("../data/test1.json")
predict_to_file("../data/test2.json")
#predict_to_file("../data/demo/demo_dev.json")
