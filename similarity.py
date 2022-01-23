import numpy as np 
import cupy as cp 
import hdbscan 
import os 
from shutil import copy2 


from glob import glob 
from tqdm import tqdm 
from cuml import HDBSCAN


# /rapids/notebooks/host

# sudo docker run --gpus all -v /home/user/faceid:/rapids/notebooks/host/faceid --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 rapidsai/rapidsai-core:21.12-cuda11.4-runtime-ubuntu20.04-py3.8

"""
!conda install hdbscan 
"""


emb_files = glob( '/rapids/notebooks/host/faceid/embeddings/**/*.np', recursive=True)
embeddings = np.array([ np.fromfile(x, dtype=np.float32) for x in tqdm(emb_files) ])


# embeddings = [ np.load(x, dtype=np.float32) for x in tqdm(embeddings) ]
model = HDBSCAN(min_samples=25, min_cluster_size=25, cluster_selection_epsilon=15.)
labels = model.fit_predict(embeddings) 


transform_fname = lambda x : x.replace('embeddings', 'face_thumbnails').replace('.np','.jpg')
thumbnails = [ transform_fname(x) for x in emb_files ]


basepath = '/rapids/notebooks/host/faceid/clusters/'
for label, thumbnail in zip ( labels, thumbnails): 
    if label < 0 : 
        continue
    try : 
        _ = os.mkdir( basepath+str(label) )
    except FileExistsError as e: 
        pass 
    if os.path.exists(thumbnail) : 
        _ = copy2(thumbnail , basepath+str(label)+'/')


