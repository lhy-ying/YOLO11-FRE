import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/trunk/exp/weights/best.pt') # select your model.pt path
    model.predict(source=r'G:\树干识别模型\data\val\images',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  classes=0,
                )