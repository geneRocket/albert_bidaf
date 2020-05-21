import json, os
import random
import torch

def load_data(filename):
    D = []
    for d in json.load(open(filename))['data'][0]['paragraphs']:
        for qa in d['qas']:
            D.append([
                qa['id'], d['context'], qa['question'],
                [a['text'] for a in qa.get('answers', [])]
            ])
    random.shuffle(D)
    return D

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class DataGenerator():
    def __init__(self,json_path,tokenizer,device):
        self.train_data = load_data(json_path)
        print(len(self.train_data))
        self.tokenizer=tokenizer
        self.device=device


    def batchIter(self,batch_size):
        batch_p=[]
        batch_q=[]
        batch_start=[] #答案在context的下标
        batch_end=[]
        batch_context_len=[]
        batch_question_len=[]

        batch_ans_ids=[]
        batch_answers=[]

        for cnt,(id,context,question,answers) in enumerate(self.train_data):
            max_len=240
            if(len(context)>max_len):
                context=context[:max_len]

            context_ids=self.tokenizer.encode(context)[1:-1]
            question_ids=self.tokenizer.encode(question)[1:-1]
            answer=random.choice(answers)
            ans_ids = self.tokenizer.encode(answer)[1:-1]

            has_answer = search(ans_ids, context_ids)

            if has_answer == -1:
                continue
            batch_p.append(context)
            batch_q.append(question)
            batch_ans_ids.append(ans_ids)
            batch_answers.append(answer)
            batch_context_len.append(len(context_ids))
            batch_question_len.append(len(question_ids))
            
            if len(batch_p)>=batch_size or cnt==len(self.train_data)-1:

                ret=self.tokenizer.batch_encode_plus(zip(batch_p,batch_q),pad_to_max_length=True)
                batch_pair_ids=ret['input_ids']
                batch_token_type_ids=ret['token_type_ids']
                batch_attention_mask=ret['attention_mask']

                for i in range(len(batch_pair_ids)):
                    start_idx=search(batch_ans_ids[i], batch_pair_ids[i])-1 #[cls]para[sep]ques[sep],后面需要拆开para和ques,所以以para为参考
                    batch_start.append(start_idx)
                    batch_end.append(start_idx+len(batch_ans_ids[i])-1)


                batch_pair_ids = torch.tensor(batch_pair_ids).to(self.device)
                batch_token_type_ids = torch.tensor(batch_token_type_ids).to(self.device)
                batch_attention_mask = torch.tensor(batch_attention_mask).to(self.device)
                batch_start = torch.tensor(batch_start).to(self.device)
                batch_end = torch.tensor(batch_end).to(self.device)
                
                yield {
                    "batch_pair_ids": batch_pair_ids,
                    "batch_token_type_ids": batch_token_type_ids,
                    "batch_attention_mask": batch_attention_mask,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "batch_context_len":batch_context_len,
                    "batch_question_len":batch_question_len,
                    "batch_answers":batch_answers,
                    "batch_p":batch_p,
                }
                batch_p=[]
                batch_q=[]
                batch_start=[]
                batch_end=[]
                batch_context_len=[]
                batch_question_len=[]
                batch_ans_ids=[]
                batch_answers=[]

    def batchIterNoAnswer(self, batch_size,ignore_id=set()):
        batch_id=[]
        batch_p=[]
        batch_q=[]
        batch_context_ids=[]
        batch_context_len=[]
        batch_question_len=[]


        for cnt,(id,context,question,_) in enumerate(self.train_data):
            if(id in ignore_id):
                continue

            max_len=240
            if(len(context)>max_len):
                context=context[:max_len]

            context_ids=self.tokenizer.encode(context)[1:-1]
            question_ids=self.tokenizer.encode(question)[1:-1]

            batch_id.append(id)
            batch_p.append(context)
            batch_q.append(question)
            batch_context_len.append(len(context_ids))
            batch_question_len.append(len(question_ids))

            
            batch_context_ids.append(context_ids)


            if len(batch_p)>=batch_size or cnt==len(self.train_data)-1:

                ret=self.tokenizer.batch_encode_plus(zip(batch_p,batch_q),pad_to_max_length=True)
                batch_pair_ids=ret['input_ids']
                batch_token_type_ids=ret['token_type_ids']
                batch_attention_mask=ret['attention_mask']



                batch_pair_ids = torch.tensor(batch_pair_ids).to(self.device)
                batch_token_type_ids = torch.tensor(batch_token_type_ids).to(self.device)
                batch_attention_mask = torch.tensor(batch_attention_mask).to(self.device)
                
                yield {
                    "batch_id":batch_id,
                    "batch_pair_ids": batch_pair_ids,
                    "batch_token_type_ids": batch_token_type_ids,
                    "batch_attention_mask": batch_attention_mask,
                    "batch_context_ids":batch_context_ids,
                    "batch_context_len":batch_context_len,
                    "batch_question_len":batch_question_len,
                    "batch_p":batch_p,
                }
                batch_p=[]
                batch_q=[]
                batch_id=[]
                batch_context_ids=[]
                batch_context_len=[]
                batch_question_len=[]


def test():
    from transformers import BertTokenizer
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained = 'albert_chinese_tiny'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    data_gen=DataGenerator("../data/demo/demo_train.json",tokenizer,device)

    for batch in data_gen.batchIter(2):
        batch_start=batch["batch_start"]
        batch_end=batch["batch_end"]
        batch_pair_ids=batch["batch_pair_ids"]
        batch_answers=batch["batch_answers"]
        batch_p=batch["batch_p"]

        for i in range(len(batch_start)):
            start=batch_start[i].item()
            end=batch_end[i].item()
            decode_answer=tokenizer.decode(batch_pair_ids[i][start+1:end+1+1]).replace(" ","")
            if(decode_answer!=batch_answers[i]):
                print(batch_answers[i])
                print(decode_answer)
                print("===========")

    print("************************")

    for batch in data_gen.batchIterNoAnswer(2):
        batch_pair_ids=batch["batch_pair_ids"]
        batch_token_type_ids=batch["batch_token_type_ids"]
        batch_p=batch["batch_p"]
        batch_context_ids=batch["batch_context_ids"]
        
        for i in range(len(batch_pair_ids)):
            print(tokenizer.decode(batch_pair_ids[i]))
            print(tokenizer.decode(batch_context_ids[i]))
            print("===============")

    

if __name__ == "__main__":
    test()
