from ultralytics import YOLO

model_path = r'./trained-inference-graphs/yolo/runs/detect/train/weights/best.pt'



class yolo_custom():
    def __init__(self,threshold=0.4) -> None:
        
        self.model = YOLO(model_path)
        return None 

    
    def run_test_image(self, img_path ,threshold= 0.4):
        
        results = self.model(source = img_path, conf = threshold, save = True)
        
        return results