import cv2
import numpy as np
from skimage import transform
from ClassificationModels.ModelLoader import * 
from ClassificationModels.Settings import * 

def traffic_sign_prediction(image):
  traffic_light_image = cv2.resize (image, (30,60))
  traffic_light_image = np.array(traffic_light_image)
  img_batch = np.expand_dims(traffic_light_image,0)
  prediction = Trafic_Lights_model.predict(img_batch)
  prediction = np.argmax(prediction, axis=1)
  if prediction[0] == 3:p = 'yellow_traffic_light_signal'
  if prediction[0] == 1:p = 'red_traffic_light_signal'
  if prediction[0] == 2:p = 'green_traffic_light_signal' 
  if prediction[0] == 0:p = 'back_traffic_light'
  return p

def road_line_prediction(image):
  road_image = image[300:450, 0:250]
  road_image = cv2.cvtColor(road_image,cv2.COLOR_BGR2RGB) ## переводим в оттенки серого
  road_image = cv2.resize(road_image, (256,256))

  road_image = np.array(road_image)
  img_batch = np.expand_dims(road_image,0)
  prediction = np.argmax(Road_Line_model.predict(img_batch), axis=1)
  if prediction[0] == 0:p = 'Botts-dots '
  if prediction[0] == 1:p = 'continuous'
  if prediction[0] == 2:p = 'continuous_yellow'
  if prediction[0] == 3:p = 'dashed'
  if prediction[0] == 4:p = 'double_continuous'
  if prediction[0] == 5:p = 'double_dashed'

  return p

def traffic_signs_prediction(image):
  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) ## переводим в оттенки серого
  image_resized = transform.resize(image, (60,60))
  img_batch = np.expand_dims(image_resized,0)

  # Начинаем предсказание категории знака
  categorical_prediction = Trafic_Sign_model.predict(img_batch)
  categorical_prediction = np.argmax(categorical_prediction, axis=1)
  # Начинаем предсказание знака внутри категории
  if categorical_prediction[0] == 2:
      image_resized = transform.resize(image, (60,90))
      img_batch = np.expand_dims(image_resized,0)
      
      add_signs_prediction = add_signs_model.predict(img_batch)
      add_signs_prediction = np.argmax(add_signs_prediction, axis=1)
      
      return add_signs_classes[str(add_signs_prediction[0])]
      
  if categorical_prediction[0] == 0:
      forb_signs_prediction = forb_signs_model.predict(img_batch)
      forb_signs_prediction = np.argmax(forb_signs_prediction, axis=1)
      
      return  forb_signs_classes[(str(forb_signs_prediction[0]))]

  if categorical_prediction[0] == 3:
      inf_signs_prediction = inf_signs_model.predict(img_batch)
      inf_signs_prediction = np.argmax(inf_signs_prediction, axis=1)
      
      return  inf_signs_classes[(str(inf_signs_prediction[0]))]
            
  if categorical_prediction[0] == 4:
      presc_signs_prediction = presc_signs_model.predict(img_batch)
      presc_signs_prediction = np.argmax(presc_signs_prediction, axis=1)
      return  presc_signs_classes[(str(presc_signs_prediction[0]))]
      
  if categorical_prediction[0] == 5:
      image_resized = transform.resize(image, (70,70))
      img_batch = np.expand_dims(image_resized,0)
      
      priority_signs_prediction = priority_signs_model.predict(img_batch)
      priority_signs_prediction = np.argmax(priority_signs_prediction, axis=1)
      
      return  priority_signs_classes[(str(priority_signs_prediction[0]))]

  if categorical_prediction[0] == 6:
      image_resized = transform.resize(image, (120,80))
      img_batch = np.expand_dims(image_resized,0)
      
      service_sign_prediction = service_signs_model.predict(img_batch)
      service_sign_prediction = np.argmax(service_sign_prediction, axis=1)
      
      return service_signs_classes[(str(service_sign_prediction[0]))]
         
  if categorical_prediction[0] == 7:

      special_instructions_signs_prediction = special_instructions_signs_model.predict(img_batch)
      special_instructions_signs_prediction = np.argmax(special_instructions_signs_prediction, axis=1)
      
      return  special_instructions_signs_classes[(str(special_instructions_signs_prediction[0]))]

          
  if categorical_prediction[0] == 1:
      Warning_signs_prediction = Warning_signs_model.predict(img_batch)
      Warning_signs_prediction = np.argmax(Warning_signs_prediction, axis=1)
      
      return Warning_signs_classes[(str(Warning_signs_prediction[0]))]

def road_type_prediction(image):
  road_image = cv2.resize(image[300:450, 0:550], (240,360))
  road_image = cv2.cvtColor(road_image,cv2.COLOR_BGR2RGB) ## переводим в оттенки серого
  img_batch = np.expand_dims(road_image,0)
  road_class = np.argmax(Road_Type_model.predict(img_batch), axis=1)
 
  # Начинаем предсказание знака внутри категории
  if road_class[0] == 0:
      Road_Type_Snow_prediction = Road_Type_Snow_model.predict(img_batch)
      Road_Type_Snow_prediction = np.argmax(Road_Type_Snow_prediction, axis=1)
      
      if Road_Type_Snow_prediction[0] ==0: p = 'fresh_snow'
      if Road_Type_Snow_prediction[0] ==1: p = 'ice'
      if Road_Type_Snow_prediction[0] ==2: p = 'melted_snow'

      return p
  if road_class[0] == 1:
      Road_Type_Mud_prediction = Road_Type_Mud_model.predict(img_batch)
      Road_Type_Mud_prediction = np.argmax(Road_Type_Mud_prediction, axis=1)

      if Road_Type_Mud_prediction[0] ==0: p = 'dry_mud'
      if Road_Type_Mud_prediction[0] ==1: p = 'water_mud'
      if Road_Type_Mud_prediction[0] ==2: p = 'wet_mud'
      
      return p
  
  if road_class[0] == 2:
    Road_Type_Gravel_prediction = Road_Type_Gravel_model.predict(img_batch)
    Road_Type_Gravel_prediction = np.argmax(Road_Type_Gravel_prediction, axis=1)

    if Road_Type_Gravel_prediction[0] ==0: p = 'dry_gravel'
    if Road_Type_Gravel_prediction[0] ==1: p = 'water_gravel'
    if Road_Type_Gravel_prediction[0] ==2: p = 'wet_gravel'
    
    return p
  if road_class[0] == 3:
    Road_Type_Concrete_prediction = Road_Type_Concrete_model.predict(img_batch)
    Road_Type_Concrete_prediction = np.argmax(Road_Type_Concrete_prediction, axis=1)
    
    if Road_Type_Concrete_prediction[0] ==0: p = 'dry_concrete'
    if Road_Type_Concrete_prediction[0] ==1: p = 'water_concrete'
    if Road_Type_Concrete_prediction[0] ==2: p = 'wet_concrete'
    
    return p
      
  if road_class[0] == 4:
      Road_Type_Asphalt_prediction = Road_Type_Asphalt_model.predict(img_batch)
      Road_Type_Asphalt_prediction = np.argmax(Road_Type_Asphalt_prediction, axis=1)
      if Road_Type_Asphalt_prediction[0] ==0: p = 'dry_asphalt'
      if Road_Type_Asphalt_prediction[0] ==1: p = 'water_asphalt'
      if Road_Type_Asphalt_prediction[0] ==2: p = 'wet_asphalt'
      return  p