import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'  

import pandas as pd 
import numpy as np 
import faiss

# user_emb_path = "user_emb.parquet"
# df_user = pd.read_parquet(user_emb_path)

query_emb_path = "query_emb.parquet"

df_query = pd.read_parquet(query_emb_path)
user_emb = np.array([list(value) for value in df_query['embedding'].values]).reshape(-1,4096)


album_emb_paths = [f"album_emb_{i+1}.parquet" for i in range(9)]
df_album_list = []
for album_emb_path in album_emb_paths:
    df_album_list.append(pd.read_parquet(album_emb_path))
df_album = pd.concat(df_album_list)
album_emb = np.array([list(value) for value in df_album['embedding'].values]).reshape(-1,4096)

dim = 4096
index_type = 'Flat'
metric_type = faiss.METRIC_INNER_PRODUCT
index = faiss.index_factory(dim,index_type,metric_type)
#index = faiss.index_factory(d, "IVF4096_HNSW32,Flat", metric_type)

index.train(album_emb)
index.add(album_emb)

print('index.ntotal=',index.ntotal,'\n')   

k = 20                     # topK的K值
D, I = index.search(user_emb, k)

for i in range(I.shape[0]):
    idx_list = list(I[i])
    # userId = df_user['userId'][i]
    input = df_query['input'][i]
    albumIds = df_album['albumId'].iloc[idx_list].values
    # print(f"userId: {userId}, albumIds: {albumIds}")
    print("list album: ",",".join(albumIds))
    print(f"query: {input}, albumIds: {albumIds}")
    

