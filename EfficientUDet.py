from SemanticModel.Head import *
from SemanticModel.Neck import *
from SemanticModel.Backbone import *
from keras import models

class EfficientUDet():
    def __init__(self, num_classes) -> None:
        self.num_classes=num_classes
        pass
    
    def Semantic_model(self, phi, num_anchors=9, freeze_bn=False,
                    score_threshold=0.01, detect_quadrangle=False, anchor_parameters=None, separable_conv=True):
        
        ############ build Backbone ############
        print(phi)
        assert phi in range(8)
        image_input, w_bifpn, d_bifpn, w_head, d_head, fpn_features = EfficientNet(img_size = 640).get_efnet(phi)

        ############ build wBiFPN ############
        for i in range(d_bifpn):
            fpn_features = wBiFPN().build_wBiFPN(fpn_features, w_bifpn, i, freeze_bn=freeze_bn)

        ########### Create Heads ############
        u_net = UNet.build_unet(fpn_features, self.num_classes)

        ############ Assemble Model ############
        model = models.Model(image_input,u_net, name='EfficientUDet') 
        return model
