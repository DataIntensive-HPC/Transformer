import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm





def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

def factorized_attention(q_I, W_A, W_B, W_Bt, W_At, qt, v, d_k, mask=None, dropout=None):

    #Left To Right Operation

    # Main part of the self attention is Q*K^T which can be represented as I * W_A * W_B * W_Bt *W_At * I_^t

    # I should reshape Input q_I
    #bs=q_I.size(0)

    #q_I = 
    print("q_I matrix size")
    print(q_I.size())

    print("WA matrix size")
    print(W_A.size())

    #Calculate I * A
    IA = torch.einsum('kabc,bcj->kabj', [q_I, W_A] )
 
    print("IA Matrix size")
    print(IA.size())

    #Calculate IA * B
    IAB = torch.einsum('kabj,bji->kabi', [IA, W_B] )

    print("IAB Matrix size")
    print(IAB.size())


    #Calculate IAB*Bt
    IABBt = torch.einsum('kabi,bim->kabm', [IAB, W_Bt])

    print("IABBt Matrix size")
    print(IABBt.size())

    #Calculate IABBt * At
    IABBtAt = torch.einsum('kabm,bmj->kabj' , [IABBt , W_At])

    print("IABBtAt Matrix size")
    print(IABBtAt.size())

    #Calculate I^T
    #It = q_I.transpose(-2, -1)

    scores = torch.einsum('kabj,kbjm->kabm' , [IABBtAt, qt])

    print("score")
    print(scores.size())

    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    print(" v size")
    print(v.size())    
    #output = torch.matmul(scores, v)
    output = torch.einsum('kabm,kbmj->kbaj', [scores, v])
    return output

 

class FactorizedMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, factorized_k=32, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads


        self.h = heads
        self.factorized_k = factorized_k
        self.d_fk= factorized_k // heads

        #self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        #self.k_linear = nn.Linear(d_model, d_model)
        
        # W1 = A * B Factorized Version of W for Q
        # W2 = A2 * B2 Factorized Version of W for K

        # Factorized Weight Matrix for Q 
        self.W_A = nn.Parameter(torch.rand(d_model, factorized_k), requires_grad=True)
        self.W_B = nn.Parameter(torch.rand(factorized_k, d_model), requires_grad=True)

        # Factorized Weight Matrix for K
        self.W_A2 = nn.Parameter(torch.rand(d_model, factorized_k), requires_grad=True)
        self.W_B2 = nn.Parameter(torch.rand(factorized_k, d_model), requires_grad=True)

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(d_model, d_model)
    

    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        print("bs is ")
        print(bs)

        print("q size is")
        print(q.size())

        # perform linear operation and split into N heads

        #k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        #q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        #k = k.transpose(1,2)
        #q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        #scores = attention(q, k, v, self.d_k, mask, self.dropout)

        #self.W_A = self.W_A.view(bs, -1, self.h, self.d_fk)
        #self.W_B = self.W_B.view(bs, -1, self.h, self.d_fk)
        W_a = self.W_A.view(self.h, self.d_k,-1)
        W_b = self.W_A.view(self.h, -1, self.d_k)

        #self.W_A2 = self.W_A2.view(bs, -1, self.h, self.d_fk)
        #self.W_B2 = self.W_B2.view(bs, -1, self.h, self.d_fk)
        W_a2 = self.W_A2.view(self.h, self.d_k,-1)
        W_b2 = self.W_A2.view(self.h, -1, self.d_k)


        qt = torch.einsum("abc->acb", [q])
        #qt = qt.view(bs, -1, self.d_k)
        qt = qt.view(bs, self.h, self.d_k, -1)

        print("qt size is")
        print(qt.size())

        q =q.view(bs, -1, self.h, self.d_k)



        scores = factorized_attention(q, W_a, W_b, W_a2, W_b2 ,qt, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
