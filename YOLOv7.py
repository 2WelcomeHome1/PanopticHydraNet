import torch
import cv2
import yaml
import numpy as np

from PIL import Image
from torchvision import transforms
from InstanceModel.utils.datasets import letterbox
from InstanceModel.utils.general import non_max_suppression_mask_conf
from detectron2.modeling.poolers import ROIPooler


class YOLOSettings:
    def __init__(self, device) -> None:
        self.device = device
        pass

    def run_inference(self, url, model):
        image = cv2.imread(url)
        image = letterbox(image, 640, stride=64, auto=True)[0] 
        image = transforms.ToTensor()(image)
        image = image.half().to(self.device) 
        image = image.unsqueeze(0)
        output = model(image)
        return output, image
    
    def re_parammetatizing(self, inf_out, attn, bases, sem_output, image, model):
        bases = torch.cat([bases, sem_output], dim=1)
        nb, _, height, width = image.shape
        names = model.names
        pooler_scale = model.pooler_scale

        with open('./InstanceModel/data/hyp.scratch.mask.yaml') as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)

        pooler = ROIPooler(output_size=hyp['mask_resolution'], 
                            scales=(pooler_scale,), 
                            sampling_ratio=1, 
                            pooler_type='ROIAlignV2', 
                            canonical_level=2)
                            
        # output, output_mask, output_mask_score, output_ac, output_ab
        output, output_mask, _, _, _ = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp,
                                                                    conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None) 
        return  output, output_mask, height, width, names

    def load_model(self, weights_path):
        model = torch.load(weights_path, map_location=self.device)['model']
        model.eval()

        if torch.cuda.is_available(): model.half().to(self.device)
        return model

    def crop_image(self, x1,y1,x2,y2, first_image):
        im = Image.fromarray(first_image)
        img2 = im.crop((x1,y1,x2,y2))
        img = np.array(img2)
        return img






