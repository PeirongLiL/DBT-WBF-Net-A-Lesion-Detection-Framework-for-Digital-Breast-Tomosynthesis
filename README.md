# Enhanced Mass and Architectural Distortion Detection in Digital Breast Tomosynthesis via Cross-Slice Weighted Box Fusion

This repository contains the codebase for the breast cancer detection and evaluation framework presented in our paper (Li, P., Li, X., Shi, P. (2026). DBT-WBF: A Weighted Boxes Fusion Model for Improving Breast Cancer Detection in DBT. Smart Innovation, Systems and Technologies, vol 444. Springer, Singapore. https://doi.org/10.1007/978-981-96-7273-8_23). It utilizes YOLOv8 for detection and incorporates a Detection Box Fusion method (DBT_WBF) proposed in the study to enhance final prediction results. The framework includes dataset preparation, training, evaluation using AUC and FROC metrics, and fusion of model outputs.

config.py -- Contains configuration settings and parameters for training and inference using YOLOv8. We trained YOLO using the mmdet library (https://mmyolo.readthedocs.io/zh-cn/latest/).

Generate_training_sets.py -- Script for generating the training dataset from annotated images and masks. BCS-DBT dataset can be found in https://sites.duke.edu/mazurowski/resources/digital-breast-tomosynthesis-database/. 

Generate_validing_sets.py -- Script for generating the validation dataset.

Generate_testing_sets.py -- Script for generating the test dataset used for evaluation.

Count_auc.py -- Evaluates model performance using ROC curves and calculates AUC (Area Under the Curve) for detection boxes.

Count_FROC.py -- Computes FROC (Free-Response Receiver Operating Characteristic) curves for detailed performance analysis, especially in medical image detection.

DBT_WBF.py -- Implements our proposed detection box fusion algorithm (DBT_WBF), which improves final detection results by integrating multiple predictions cross slices.
