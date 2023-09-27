from PanopticHydraNet import RoadSceneDetector

if __name__ == "__main__":
    img_size = 640
    num_classes = 23
    image_path = 'P1.png'
    num_classes, image_path, img_size
    RoadSceneDetector(num_classes, image_path, img_size).run()