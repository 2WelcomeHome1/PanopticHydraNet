import cv2
import numpy as np
from PIL import Image

from Models import *
from EfficientPanopticModel import PanopticModel

class RoadSceneDetector(PanopticModel):
    def __init__(self, num_classes, image_path, img_size):
        super().__init__(num_classes, image_path, img_size)
        self.image_path = image_path
        pass
    def crop_image(self, x1,y1,x2,y2, first_image):
        im = Image.fromarray(first_image)
        img2 = im.crop((x1,y1,x2,y2))
        img = np.array(img2)
        return img
    
    def panoptic_bounding(self, panoptic_image, original_image, pred_masks_np, names, pred_cls, pred_conf, image_path, nbboxes):
        height,width = original_image.shape[:2]
        img = cv2.imread(image_path)
        original_image = cv2.resize(img, (width,height))
        pred_img = cv2.addWeighted(panoptic_image.astype(np.int32), 0.8,original_image.astype(np.int32), 1.,0)
        for i in range(len(pred_masks_np)):
            pred_img = pred_img.copy()
            if pred_conf[i]>=0.5:
                color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
                pred_img = cv2.rectangle(pred_img, (nbboxes[i][0], nbboxes[i][1]), (nbboxes[i][2], nbboxes[i][3]), color, 2)
                if int(pred_cls[i]) == 9:
                    traffic_light_image = self.crop_image(nbboxes[i][0], nbboxes[i][1], nbboxes[i][2], nbboxes[i][3], original_image)
                    pred_label = traffic_sign_prediction(traffic_light_image)
                    label = '%s %.3f' % (pred_label, pred_conf[i])
                else: 
                    label = '%s %.3f' % (names[int(pred_cls[i])], pred_conf[i])
                t_size = cv2.getTextSize(label, 0, fontScale=0.1, thickness=1)[0]
                c2 = nbboxes[i][0] + t_size[0], nbboxes[i][1] - t_size[1] - 3
                pred_img = cv2.rectangle(pred_img, (nbboxes[i][0], nbboxes[i][1]), c2, color, -1, cv2.LINE_AA)
                pred_img = cv2.putText(pred_img, label, (nbboxes[i][0], nbboxes[i][1] - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        label_line = str(road_line_prediction(original_image))
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        pred_img = cv2.rectangle(pred_img, (0, 300), (250, 450), color, 2, cv2.LINE_AA)
        pred_img = cv2.putText(pred_img, label_line, (0, 300 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        label_road = str(road_type_prediction(original_image))
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        pred_img = cv2.rectangle(pred_img, (0, 300), (550, 450), color, 2, cv2.LINE_AA)
        pred_img = cv2.putText(pred_img, label_road, (400, 300 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        panoptic_bboxes_image = np.array(pred_img)
        
        return panoptic_bboxes_image

    def run(self):
        self.load_instance_model()
        self.load_semantic_model()
        pred_masks_np, pred_cls, pred_conf, \
            nimg, nbboxes, height, width, original_image, names = self.instance_model_prediction()
        pred = self.semantic_model_prediction()
        panoptic_image = self.panoptic_fusion(pred, original_image, pred_masks_np, height, width, pred_conf)
        height,width = original_image.shape[:2]
        panoptic_bboxes_image = self.panoptic_bounding(panoptic_image, original_image, pred_masks_np, names, pred_cls, pred_conf, self.image_path, nbboxes)
        cv2.imwrite('./HydraPrediction.jpg', panoptic_bboxes_image)
        pass


