import numpy as np

class skip_gram:
    def __init__(self,token_ls):
        self.token_ls=token_ls
        self.word_ls=sorted(list(set(self.token_ls)))

    def connected_token(self,token,window_size):
        token_index_ls=[i for i,v in enumerate(self.token_ls) if v==token]
        return_set={self.word_ls.index(v) for i,v in enumerate(self.token_ls) for j in token_index_ls if abs(i-j)<=window_size}-{self.word_ls.index(token)}
        return sorted(list(return_set))


L=["딸기","바나나","사과","딸기","파인애플","포도","블루베리"]

sg = skip_gram(L)
print(sorted(sg.word_ls))
sg.connected_token("딸기",2)

n=len(sg.word_ls)
X=[[np.identity(n)[i] for j in sg.connected_token(sg.word_ls[i],2)] for i in range(n)]
X=sum(X,[])
y=[[np.identity(n)[j] for j in sg.connected_token(sg.word_ls[i],2)] for i in range(n)]
y=sum(y,[])
X,y=np.array(X),np.array(y)
X_1=np.column_stack((np.ones(len(X)),X))

learning_rate=0.003
num_epoch=400000
np.random.seed(0)
hidden_num=5

W_1=np.random.random(size=(n+1,hidden_num))
W_2=np.random.random(size=(hidden_num+1,n))

for epoch in range(num_epoch):
    first_layer=np.dot(X_1,W_1)
    output_layer=np.dot(np.column_stack((np.ones(len(first_layer)),first_layer)),W_2)
    hyp=(np.exp(output_layer).T/np.exp(output_layer).sum(axis=1)).T
    error=-np.multiply(y,np.log(hyp)).sum()/len(X)
    
    len_neighbor=[len(sg.connected_token(sg.word_ls[j],2)) for j in range(n)]
    arr=np.zeros(n)
    for p,v in enumerate(np.argsort(-hyp)[[sum(len_neighbor[:j]) for j in range(n)]]):
        arr=np.row_stack((arr,np.identity(n)[v<len_neighbor[p]]))
    pred=arr[1:]
    accuracy=(y.argmax(axis=1)==pred.argmax(axis=1)).mean()
    if epoch%10000==0:
        print(f"epoch : {epoch} loss : {error}")
    if error<1.13:
        break
    W_1=W_1-np.dot(X_1.T,np.dot(hyp-y,W_2.T).T[1:].T)*learning_rate/len(X)
    W_2=W_2-np.dot(np.column_stack((np.ones(len(first_layer)),first_layer)).T,(hyp-y))*learning_rate/len(X)
print(f"epoch : {epoch}, loss : {error}")