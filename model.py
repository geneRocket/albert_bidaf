
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn import LSTM, Linear

class config:
    def __init__(self):
        self.hidden_size=2048
        #self.hidden_size=312
        self.dropout=0.1

class BertBidaf(nn.Module):
    def __init__(self,tokenizer,encode_model,device):
        super(BertBidaf, self).__init__()

        # 1. bert encode
        self.tokenizer = tokenizer
        self.encode_model = encode_model
        self.device=device

        args=config()

        self.pad_vec = torch.zeros(args.hidden_size).to(self.device).detach()

        self.att_weight_c = Linear(args.hidden_size , 1)
        self.att_weight_q = Linear(args.hidden_size , 1)
        self.att_weight_cq = Linear(args.hidden_size , 1)

        self.output_linear = Linear(args.hidden_size * 4, 2)

    def bert_encode(self,batch_pair_ids,batch_token_type_ids,batch_attention_mask,batch_context_len,batch_question_len):
        outputs = self.encode_model(batch_pair_ids,attention_mask=batch_attention_mask,token_type_ids=batch_token_type_ids)

        pair_vec = outputs[0]
        batch_size = pair_vec.size(0)

        #split
        para_vec_list = []
        question_vec_list = []
        para_max_len = 0
        ques_max_len = 0
        for i in range(batch_size):
            para_vec_list.append(pair_vec[i][1: 1 + batch_context_len[i]])
            para_max_len = max(para_max_len, batch_context_len[i])
            question_first_index=1 + batch_context_len[i] + 1
            question_vec_list.append(pair_vec[i][question_first_index:question_first_index+batch_question_len[i]])
            ques_max_len = max(ques_max_len, batch_question_len[i])

        dim = pair_vec.size(2)
        
        #pad
        for i in range(batch_size):
            len = para_vec_list[i].size(0)
            if para_max_len - len > 0:
                para_vec_list[i] = torch.cat([para_vec_list[i], self.pad_vec.expand(para_max_len - len, dim)], dim=0).to(self.device)
            len = question_vec_list[i].size(0)
            if ques_max_len - len > 0:
                question_vec_list[i] = torch.cat([question_vec_list[i], self.pad_vec.expand(ques_max_len - len, dim)], dim=0).to(self.device)

        para_vec_batch = torch.stack(para_vec_list, dim=0).to(self.device)
        ques_vec_batch = torch.stack(question_vec_list, dim=0).to(self.device)

        return para_vec_batch,ques_vec_batch

    def att_flow_layer(self,c, q,c_len_mask,q_len_mask):
        batch_size=c.size(0)
        c_len = c.size(1)
        q_len = q.size(1)

        
        cq = []
        for i in range(q_len):#遍历每一个问题的字
            qi = q.select(1, i).unsqueeze(1) #[batch,1,dim],选择每一字的向量
            ci = self.att_weight_cq(c * qi).squeeze() # c*qi: [batch,clen,dim]*[batch,1,dim] -> [batch,clen,dim] , linear: [batch,clen]  所有文章的字 和 问题中的一个字 的相关系数 
            cq.append(ci)
        cq = torch.stack(cq, dim=-1) # [batch,clen,qlen] # 文章的字 和 问题中的字 的相关系数

        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq
        
        #mask
        mask=torch.zeros((batch_size,c_len,q_len)).float().to(self.device)
        for i in range(batch_size):
            for col_idx in range(q_len_mask[i],q_len):
                for row in range(c_len):
                    mask[i][row][col_idx]=-1e12
            for row_idx in range(c_len_mask[i],c_len):
                mask[i][row_idx][0:-1]=-1e12

        s=s+mask

        a = F.softmax(s, dim=2) # 一个上下文 和 哪一个问题的字 的相关系数
        c2q_att = torch.bmm(a, q) # [batch,c_len,dim] 按照权重问题求和
        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1) # [batch,1,c_len] 文章每个字的重点系数
        q2c_att = torch.bmm(b, c).squeeze() # 总结文章向量 [batch,dim]
        q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1) #[batch,c_len,dim]

        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
        return x

    def forward(self,batch_pair_ids,batch_token_type_ids,batch_attention_mask,batch_context_len,batch_question_len):
        para_vec_batch,ques_vec_batch = self.bert_encode(batch_pair_ids,batch_token_type_ids,batch_attention_mask,batch_context_len,batch_question_len)

        out=self.att_flow_layer(para_vec_batch,ques_vec_batch,batch_context_len,batch_question_len)

        out=self.output_linear(out)


        #mask
        batch_size=out.size(0)
        for i in range(batch_size):
            out[i][batch_context_len[i]:-1]=-1e12

        
        p1,p2=out.split(1,dim=-1)
        p1=p1.squeeze(-1)
        p2=p2.squeeze(-1)

        return p1, p2