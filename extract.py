
import insightface 
import cv2 

import numpy as np 

from insightface.utils import face_align
from glob import glob 
from tqdm import tqdm 



import onnxruntime as ort
ort.set_default_logger_severity(3)

videos = glob( '/home/user/faceid/data/**/*.mp4', recursive=True)

detection = insightface.model_zoo.get_model("/home/user/faceid/models/scrfd_10g_bnkps.onnx", providers=["CUDAExecutionProvider"])
detection.prepare(1)

recognition = insightface.model_zoo.get_model("/home/user/faceid/models/glintr100.onnx", providers=["CUDAExecutionProvider"])
recognition.prepare(0)



for file in tqdm(videos) : 

    cap = cv2.VideoCapture(file)

    count = -1
    while cap.isOpened():
        try : 
            count += 1
            ret,frame = cap.read()
            if not ret : 
                break 
            if count % 5 != 0 : 
                continue
            bboxes, kpss = detection.detect(frame, input_size=(frame.shape[0] - frame.shape[0] % 32 , frame.shape[1] - frame.shape[1] % 32), max_num=-1)

            _, _ , _, _, _, folder, basename = file.split('/')
            basename = basename.replace('.mp4','')
            output_name = f'/home/user/faceid/face_thumbnails/{folder}-{basename}-{count}'
            embedding_outname = output_name.replace('face_thumbnails', 'embeddings')

            for i,bbox in enumerate(bboxes): 
                x1,y1,x2,y2,_ = bbox.astype(int)
                score = bbox[-1]

                face= frame[y1:y2, x1:x2 ].copy()

                #only append good faces 
                if score > .75 : 
                    aimg = face_align.norm_crop(frame, landmark=kpss[i], image_size=face.shape[0] )
                    embedding = recognition.get_feat(aimg).flatten()
                    np.save(embedding_outname+f'-{i}.npy', embedding)

                    try:
                        cv2.imwrite(output_name+f'-{i}.png', face )
                    except Exception as e:
                        continue
        except Exception as e: 
            continue
            
    cap.release()

