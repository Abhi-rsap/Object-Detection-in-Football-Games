import streamlit as st
from yolov8 import yolo_custom
import os
import shutil
import cv2
from PIL import Image
import pandas as pd

st.set_page_config(layout="wide")


def save_uploadedfile(uploadedfile):
    with open(os.path.join("tmp",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())

test_images_path = r'./test_images/'
predict_images_path = r'./runs/detect/predict/'

st.title('Object Detection In Football ')
if os.path.exists('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')


tab1, tab2 = st.tabs(['Prediction','Metrics'])

with tab1:
    yolo = yolo_custom()

    options =  ['.JPEG/PNG format','.MP4/AVI format']

    file_type = st.selectbox(label= 'Select File type:',options = options,index = None)

    if os.path.exists('runs'):
        shutil.rmtree('runs')

    if file_type in options:
        files = st.file_uploader("Choose a file", accept_multiple_files=True)
        
        if files is not None:
            threshold = st.slider(min_value=0.0, max_value=1.0,step= 0.1,label = "Minimum threshold for detection:",value=0.5)  
            run = st.button('Run model')

            
            if run:
                for file in files:
                    save_uploadedfile(file)
                st.success("Files Uploaded!")
                
                if file_type == '.JPEG/PNG format':
                    for file in files:
                        
                        result = yolo.run_test_image(img_path='tmp/'+file.name, threshold = threshold)
                        
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.header("Original Image")
                            st.image(test_images_path+file.name, caption= 'Original Image',use_column_width=True)
                        
                        with col2:
                            st.header("Predicted")
                            st.image(predict_images_path+file.name, caption='Predicted '+file.name, use_column_width=True)
                
                else:
                    for file in files:
                        
                        cap = cv2.VideoCapture('tmp/'+file.name)
                        i = 0
                        stframe = st.empty()
                        while cap.isOpened():
                            
                            ret, frame = cap.read()
                            if not ret:
                                break
                        
                            # Run inference on the frame
                            result = yolo.run_test_image(img_path= frame, threshold = threshold)
                            
                            processed_frame = Image.open(predict_images_path+'image0'+'.jpg')
                            i = i+1
                            # Display the processed frame
                            stframe.image(processed_frame,caption='Live Inference')
                            
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        
                        cap.release()

with tab2:
    st.title('1) YOLO V8 RESULTS - BEST MODEL')
    
    directory = 'all_models_results/'
    df = pd.read_csv(directory+ 'yolov8/train/results.csv')
    # st.dataframe(df,use_container_width=True)
    st.image(directory+ 'yolov8/train/results.png', width=1000,caption="Training Metrics")
    col3, col4 = st.columns(2)
    
    col5, col6 = st.columns(2)
    
    with col3:
        st.header('INFERENCE IMAGES ON TEST SET CAN BE FOUND IN THE DIRECTORY: all_models_results/yolov8/predict')
    
    with col5:   
        st.image(directory+ 'yolov8/train/confusion_matrix_normalized.png',use_column_width= True,caption="Confusion Matrix")
    
    with col4:
        st.image(directory+ 'yolov8/train/PR_curve.png',use_column_width= True,caption="PR Curve")
    
    
    st.title('2) MobileNet SSD v2 FPN LITE 640x640 RESULTS')
    
    col7, col8 = st.columns(2)
    
    col9, col10 = st.columns(2)
    
    with col7:
        st.image(directory+ 'mobilenet_ssd/eval_results_mobilessd.png',use_column_width= True,caption="Single checkpoint Evaluation Metrics for SSD")
    
    with col10:   
        st.image(directory+ 'mobilenet_ssd/Training_loss.png',use_column_width= True,caption="Sample Image showing Loss per epoch")
    
    
    st.title('3) Faster RCNN Inception Resnet v2 640x640 Results')
    
    col11, col12 = st.columns(2)
    
    col13, col14= st.columns(2)
    
    with col11:
        st.image(directory+ 'rcnn/training_loss_RCNN_1.png',use_column_width= True,caption="Training Loss metrics for RCNN 1")
    
    with col14:   
        st.image(directory+ 'rcnn/training_loss_RCNN_2.png',use_column_width= True,caption="Training Loss metrics for RCNN 2")
    
    
    
                           
    
