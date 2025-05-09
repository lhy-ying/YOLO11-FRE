import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/runs/Ablation Study/yolo11s+FasterNet+C2PSA-EMA+RFAConv/Faster-C2PSA-EMA-RFAConv_train/weights/best.pt')
    model.val(data=r'/data/class.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )