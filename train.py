import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('"/cfg/yolo11-Faster-C2PSA-EMA-RFAConv.yaml"') 
    model.train(data=r"/data/class.yaml",
                cache=False,
                imgsz=640,
                epochs=150,
                single_cls=False,  
                batch=4,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True,  
                project='runs/train',
                name='yolo11-Faster-C2PSA-EMA-RFAConv',
                )

