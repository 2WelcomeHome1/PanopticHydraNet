from keras import models
import tensorflow_addons as tfa

# Trafic Sign categorical model
Trafic_Sign_model = models.load_model('./ClassificationModels/Trafic_Signs_CNN_categorical.h5')

# Signs model
add_signs_model = models.load_model('./ClassificationModels/Trafic_Signs_CNN_add_signs.h5')
forb_signs_model = models.load_model('./ClassificationModels/Trafic_Signs_CNN_forb_signs.h5')
inf_signs_model = models.load_model('./ClassificationModels/Trafic_Signs_CNN_inf_signs.h5')
presc_signs_model = models.load_model('./ClassificationModels/Trafic_Signs_CNN_presc_signs.h5')
priority_signs_model = models.load_model('./ClassificationModels/Trafic_Signs_CNN_priority_signs.h5')
service_signs_model = models.load_model('./ClassificationModels/Trafic_Signs_CNN_service_signs.h5')
special_instructions_signs_model = models.load_model('./ClassificationModels/Trafic_Signs_CNN_special_instructions_signs.h5')
Warning_signs_model = models.load_model('./ClassificationModels/Trafic_Signs_CNN_Warning_signs.h5')

# Traffic Lights model
Trafic_Lights_model = models.load_model('./ClassificationModels/Trafic_Lights_CNN.h5')

# Road Line model
Road_Line_model = models.load_model('./ClassificationModels/Road_Line_CNN.h5')

# Road Type categorical model
Road_Type_model = models.load_model('./ClassificationModels/Road_Type_AllClasses_CNN.h5')

# road model
Road_Type_Asphalt_model = models.load_model('./ClassificationModels/Road_Type_Asphalt_CNN.h5')
Road_Type_Concrete_model = models.load_model('./ClassificationModels/Road_Type_Concrete_CNN.h5')
Road_Type_Gravel_model = models.load_model('./ClassificationModels/Road_Type_Gravel_CNN.h5')
Road_Type_Mud_model = models.load_model('./ClassificationModels/Road_Type_Mud_CNN.h5')
Road_Type_Snow_model = models.load_model('./ClassificationModels/Road_Type_Snow_CNN.h5')


