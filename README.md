# PitchAware: Object Detection in Football Games
This project aims to perform object detection using different SOTA models:
1) Yolov8
2) Mobilenet SSD
3) RCNN

The following are the class labels defined for this project:
1) Player
2) Ball
3) Referee

I have implemented this in three parts:
## Checkpoint 1
1) Downloading dataset from open-source
2) Image Annotation using Roboflow Annotator (free)
3) Test-train split - Now data is ready!

## Checkpoint 2
1) Training of three models mentioned above
2) Selecting best model

## Checkpoint 3
1) GUI using streamlit
2) Showing model inferences for images/videos
3) Showing evaluation results fro all models

For training or to use SSD/RCNN, you need to have tensorflow, object-detection-api installed. Please check references for help.

Requirements which worked for me:
Python 3.9, TensorFlow 2.10.1


HOW TO RUN GUI:
Open terminal with "Checkpoint 3" directory and type the following command:
```bash
    streamlit run checkpoint-3.py
```

![image](https://github.com/user-attachments/assets/4b0ffd65-55bc-4139-b706-ff823f00bc79)
![image](https://github.com/user-attachments/assets/d63bc4ff-4871-4d59-95e2-1a3792f0d2e0)
![image](https://github.com/user-attachments/assets/7b3a9206-3f71-47c8-b96d-cae848fa4faf)
![image](https://github.com/user-attachments/assets/b36c0daa-10be-40e6-aef1-5ba517fdbcdb)
![image](https://github.com/user-attachments/assets/021a899f-dae9-47cd-a95a-94f81f596268)







## References
1) https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html 
2) https://github.com/tensorflow/models 
3) https://docs.streamlit.io/ 
4) https://neptune.ai/blog/tensorflow-object-detection-api-best-practices-to-training-evaluation-deployment 
