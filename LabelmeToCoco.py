import os 
import json
import numpy as np
import PIL.Image
from PIL import Image, ImageDraw

path="path/to/all/annotation_files" #this is path to your all annotation file(.json)
img_id=1
data_dict=[]

def polygons_to_mask(img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

def mask2box(mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]



label_list=[]
for filename in os.listdir(path):
    records={}
    f = os.path.join(filename)
    if os.path.isfile(f) and f[-4:]=="json":
        with open(f) as file:
            imgs = json.load(file)
            
            file_name=imgs["imagePath"]
            height=imgs["imageHeight"]
            width=imgs["imageWidth"]
            
            records["file_name"]=file_name
            records["image_id"]=img_id
            records["height"]=height
            records["width"]=width
            
            for shape in imgs["shapes"]:
                
                label=shape["label"]
                if label not in label_list:
                    label_list.append(label)
                category_id=label_list.index(label)
                
                olist=[]
                polygons = shape["points"]
                mask = polygons_to_mask([height,width], polygons)
                bbox=mask2box(mask)
                segmentations=[list(np.asarray(polygons).flatten())]
                obj={
                    'bbox':bbox,
                    #'bbox_mode':BoxMode.XYXY_ABS,
                    'bbox_mode':1,
                    'segmentation':segmentations,
                    'category_id':category_id,
                    }
                olist.append(obj)
            records["annotations"]=olist        
            img_id+=1
            data_dict.append(records)

for i in data_dict:
    print(i,"\n")
                    


