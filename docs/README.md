
# YOLO11-FRE: Enhanced Object Detection for Classroom Student Behavior Recognition

This repository contains the source code for the YOLO11-FRE model, which is proposed in the paper "Enhanced YOLO11-FRE: Real-Time and Accurate Student Behavior Recognition in Educational Surveillance" (submitted to *The Visual Computer*).

## 🌟 Highlights
- **Model Innovation**:
  - The backbone network is replaced with FasterNet, improving feature extraction efficiency while reducing computational load.
  - The RFAConv module is incorporated in the Neck layer to enhance multi-scale feature fusion.
  - The EMA attention mechanism is embedded in the C2PSA structure to focus on key behavior features and improve recognition accuracy.
- **Performance Improvement**:
  - Achieves an mAP50 of 86.7% (3.4% increase compared to the original YOLO11) on a self-constructed classroom behavior dataset.
  - Improves accuracy and recall by 1.7% and 3.4% respectively.
  - Balances accuracy and real-time performance, making it suitable for intelligent education surveillance scenarios.

## 📊 Model Architecture
![YOLO11-FRE Architecture](path/to/architecture-diagram.png)
(Describe the core modules: FasterNet backbone, RFAConv module, EMA attention mechanism)

## 📅 Dataset
This study uses a self-constructed **Classroom Student Behavior Dataset**, which includes:
- **Data Scale**: N classroom video clips, M annotated images, covering K types of student behaviors (such as listening, raising hands, writing, bowing heads, talking, etc.).
- **Annotation Format**: COCO format, including bounding boxes and behavior category labels.
- **Application Method**:
  The dataset is currently not publicly available and is only open for academic research applications. To obtain it, please send an application email to **20231800129@imut.edu.cn** with the subject line: **[Dataset Application] YOLO11-FRE Classroom Behavior Dataset**. The content should include:
  1. Applicant's name, institution, and contact information (email/phone).
  2. Research purpose and usage (specify non-commercial use).
  3. Commitment to abide by data usage regulations and not disclose or spread the dataset.

## 🚀 Code Structure
```
YOLO11-FRE/
├── cfg/                # Model configuration files (FasterNet, RFAConv, EMA module definitions)
├── data/               # Dataset configuration and preprocessing scripts
├── utils/              # Utility functions (training/testing/visualization)
├── train.py            # Training script
├── val.py              # Validation script
├── README.md           # Project description
└── LICENSE             # Open source license (recommended to use MIT/AGPL, etc.)
```

## 🛠️ Usage
### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Model
```bash
python train.py --data data/class.yaml --cfg cfg/yolov11s.yaml --weights '' --epochs 150
```

### 3. Testing and Inference
```bash
python val.py --data data/class.yaml --cfg cfg/yolov11s.yaml --weights runs/train/exp/weights/best.pt
```

## 📝 Citation
If you use the code or dataset of this project in your research, please cite the following paper:
```bibtex
@article{your-paper-title,
  title={Enhanced YOLO11-FRE: Real-Time and Accurate Student Behavior Recognition in Educational Surveillance},
  author={Your Name},
  journal={The Visual Computer},
  year={2025},
  note={DOI: To be assigned (forthcoming)},
  code={https://github.com/your-username/YOLO11-FRE},
  dataset={Application access: 20231800129@imut.edu.cn}
}
```

## 📧 Dataset Application Email Template
**Subject**: [Dataset Application] YOLO11-FRE Classroom Behavior Dataset - [Your Name/Institution]

Dear Developer,

I am [Your Name] from [Institution/School/Organization], currently engaged in research related to [Research Area, such as intelligent education, computer vision]. We are interested in your open-source project **YOLO11-FRE** (https://github.com/your-username/YOLO11-FRE) and would like to apply for the use of your self-constructed dataset for our research on **[Specific Research Purpose, e.g., "Optimization of Classroom Attention Analysis Algorithm"]**.

### Application Information:
1. **Applicant Information**:
   - Name: [Your Full Name]
   - Institution: [Institution Name]
   - Email: [Your Email]
   - Phone: [Optional]
2. **Research Purpose and Usage**:
   [Briefly describe the research objectives and experimental design, and clearly state that the data will only be used for academic research and not for commercial purposes].
3. **Data Usage Commitment**:
   We promise to abide by the data usage regulations, not disclose or spread the dataset, and clearly indicate the data source and cite your paper when publishing research results.

### Contact Information:
Please feel free to contact me at [Your Email/Phone] if you need further communication.
Thank you for your contribution to the open-source community! Looking forward to your reply.

Best regards,

[Your Name]
[Date]

## 📄 Note
1. **Open Source License**: Add a `LICENSE` file (such as MIT license) in the root directory of the GitHub repository to clarify the code usage rights.
2. **Dataset Description**: Supplement the specific scale, collection scenario, and annotation process of the dataset in the README (if there is an ethical review, it needs to be stated).
3. **Reproducibility**: Ensure that the code includes complete dependencies (`requirements.txt`), training/testing commands, and download links for pre-trained models (if available).

Please let me know if you have any further questions or need additional help.
