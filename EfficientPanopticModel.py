import torch
import cv2
from YOLOv7 import *
from EfficientUDet import *
from keras import metrics
from keras.optimizers import Adam
from util import get_colors
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

class PanopticModel():
  def __init__(self, num_classes:int, image_path:str, img_size:int):
    self.num_classes = num_classes
    self.image_path = image_path
    self.output_size = (img_size, 384)
    self.colors = np.array([(0, 0, 0),
                  (70, 70, 70),
                  (100, 40, 40),
                  (55, 90, 80),
                  (220, 20, 60),
                  (153, 153, 153),
                  (157, 234, 50),
                  (128, 64, 128),
                  (244, 35, 232),
                  (107, 142, 35),
                  (0, 0, 142),
                  (102, 102, 156),
                  (220, 220, 0),
                  (70, 130, 180),
                  (81, 0, 81),
                  (150, 100, 100),
                  (230, 150, 140),
                  (180, 165, 180),
                  (250, 170, 30), 
                  (110, 190, 160),
                  (170, 120, 50),
                  (45, 60, 150),
                  (145, 170, 100)])
    
    pass

  def load_instance_model(self):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.instance_model = YOLOSettings(self.device).load_model('InstanceModel/yolov7-mask.pt')

  def load_semantic_model(self):
    opt = Adam(learning_rate=0.0003)
    self.Semantic_model = EfficientUDet(num_classes=self.num_classes).Semantic_model(phi=0)
    self.Semantic_model.compile(optimizer=opt,loss='binary_crossentropy', 
                            metrics=[metrics.OneHotIoU(self.num_classes,[i for i in range(0,self.num_classes)])])
    self.Semantic_model.load_weights('SemanticModel/sem_weights.h5')

  def instance_model_prediction(self):        
      output, image = YOLOSettings(self.device).run_inference(self.image_path, self.instance_model)
      inf_out = output['test']
      attn = output['attn']
      bases = output['bases']
      sem_output = output['sem']

      output, output_mask, height, width, names = YOLOSettings(self.device).re_parammetatizing(inf_out, attn, bases, sem_output, image, self.instance_model)
      pred, pred_masks = output[0], output_mask[0]
      base = bases[0]
      bboxes = Boxes(pred[:, :4])

      with open('./InstanceModel/data/hyp.scratch.mask.yaml') as f:
          hyp = yaml.load(f, Loader=yaml.FullLoader)

      original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
      original_image = np.moveaxis(image.cpu().numpy().squeeze(), 0, 2).astype('float32')
      original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

      pred_masks = retry_if_cuda_oom(paste_masks_in_image)(original_pred_masks, bboxes, (height, width), threshold=0.5)

      pred_masks_np = pred_masks.detach().cpu().numpy()
      pred_cls = pred[:, 5].detach().cpu().numpy()
      pred_conf = pred[:, 4].detach().cpu().numpy()
      nimg = image[0].permute(1, 2, 0) * 255
      nimg = nimg.cpu().numpy().astype(np.uint8)
      nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
      nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int)
      return pred_masks_np, pred_cls, pred_conf, nimg, nbboxes, height, width, original_image, names

  def semantic_model_prediction(self):
      img = cv2.resize(cv2.imread(self.image_path), self.output_size)
      pred = self.Semantic_model.predict(np.array([img]))
      pred = np.argmax (pred, axis=-1) [0,:,:]
      return pred

  def panoptic_fusion(self, pred, original_image, pred_masks_np, height, width, pred_conf):

    semantic_image = Image.fromarray(self.colors[get_colors(pred)].astype(np.uint8))
    height,width = original_image.shape[:2]
    pred_img = semantic_image.resize((width, height))
    for i in range(len(pred_masks_np)):
      if pred_conf[i]>=0.5:
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        pred_img = np.array(pred_img).copy()
        pred_img[pred_masks_np[i]] = pred_img[pred_masks_np[i]] + np.array(color, dtype=np.uint8)
    
    panoptic_image = np.array(pred_img)
    
    return panoptic_image

  def panoptic_bounding(self, pred_img, nbboxes, names, pred_cls, pred_conf, pred_masks_np):
    for i in range(len(pred_masks_np)): 
      pred_img = np.array(pred_img).copy()
      color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
      pred_img = cv2.rectangle(pred_img, (nbboxes[i][0], nbboxes[i][1]), (nbboxes[i][2], nbboxes[i][3]), color, 2)
      label = '%s %.3f' % (names[int(pred_cls[i])], pred_conf[i])
      t_size = cv2.getTextSize(label, 0, fontScale=0.1, thickness=1)[0]
      c2 = nbboxes[i][0] + t_size[0], nbboxes[i][1] - t_size[1] - 3
      pred_img = cv2.rectangle(pred_img, (nbboxes[i][0], nbboxes[i][1]), c2, color, -1, cv2.LINE_AA)
      pred_img = cv2.putText(pred_img, label, (nbboxes[i][0], nbboxes[i][1] - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
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

    new_img = cv2.resize(cv2.imread(self.image_path), (width,height))
    p_image = cv2.addWeighted(panoptic_image.astype(np.int32), 0.8,new_img.astype(np.int32), 1.,0)
    panoptic_bboxes_image = self.panoptic_bounding(p_image, nbboxes, names, pred_cls, pred_conf, pred_masks_np)
    cv2.imwrite('./PanopticPrediction.jpg', panoptic_bboxes_image)


