# Project Write-Up

## Explaining Custom Layers

According to OpenVINO Toolkit >> Custom Layer Extensions for the Inference Engine:
"Each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. 
The custom layer extension is implemented according to the target device."

"Custom Layer CPU Extension - is a compiled shared library (.so or .dll binary) needed by the CPU Plugin for executing the custom layer on the CPU."

"Layer — The abstract concept of a math function that is selected for a specific purpose (relu, sigmoid, tanh, convolutional)." 

I use in my app the AddExtension method to load the extensions when for models featuring layers from Custom Layer CPU Extension library.
I use my check_layers.py to see the layers that are supported by device plugin for the Inference Engine.

## Assess Model Use Cases

Some potential use cases of the people counter application are security in places such as the Helth Center or airports, statistics for retailers, shopping centers, public transport, police and traffic management.

Each of that use cases would be useful because it could improve organization and management in these places, and also use resources more efficiently. Where security is concerned, the application could inform security e.g. how many people are still on the premises; how many people entered the museum and how many left before closing time etc. Where business is concerned e.g.: how many people were admitted to the concert vs. how many tickets have been sold.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model.

## Comparing Model Performance
I've used Udacity workspace and OpenVINO environment and my app uses Intel IE so for start inference it needed model IR. Below I compare few models after conversion to IR.

## Model Research

In investigating potential people counter models, I've tried each of the following three models:

- Model 1: [ONNX MODEL tiny-yolov2]
  - [Model Source: https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2/model]
  - I converted the model to an Intermediate Representation with the following argument: --input_model model.onnx
  - Model Optimizer ERROR: Cannot pre-process ONNX graph after reading from model file "/home/workspace/models/tiny_yolov2/model.onnx". File is corrupt or has unsupported format. 
Details: Reference to image is not satisfied. A node refer not existing data tensor. ONNX model is not consistent.
  
- Model 2: [ONNX MODEL bvlc_alexnet]
  - [Model Source: https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_alexnet.tar.gz]
  - I converted the model to an Intermediate Representation with the following argument: --input_model model.onnx
  - The model was insufficient for the app because: output blobs: ['prob_1'], output shape: (1, 1000), not fit to app's process_output_bb() def; need shape [1x1xNofBoxesx7]

- Model 3: [CAFFE MODEL SqueezeNet_v1.1]
  - [Model Source: https://github.com/forresti/SqueezeNet]
  - I converted the model to an Intermediate Representation with the following arguments: --input_model squeezenet_v1.1.caffemodel --input_proto deploy.prototxt
  - The model was insufficient for the app because: output blob: ["prob"], output shape: (10, 1000, 1, 1) not fit to app's process_output_bb() def; need shape [1x1xNofBoxesx7]

- Model 4: [TF MODEL faster_rcnn_resnet101_coco]
  - [Model source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]
  - I converted the model to an Intermediate Representation with the following arguments: --input_model frozen_inference_graph.pb --tensorflow_use_custom_operations_config $M_O/extensions/front/tf/faster_rcnn_support.json  --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channel
  - The model was insufficient for the app because: INFO:root:Processing frames...   model_shape [1, 3, 600, 600]   Segmentation fault (core dumped)

- Model 5: [TF MODEL ssdlite_mobilenet_v2]
  - [Model source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]
    - running app on a single image:
  
          Time of Inference on 1 frame: 0.0403   
          Time of Processing output on 1 frame: 0.001092  
          INFO:root:Size of frame sent to the server: (453, 883, 3)    
          run_time: 0.44sec    
          infer_time: 0.04sec B=0.04, S=0.00
          processing_outputs_time: 0.0011sec
    
    - running app on a short video with 2 persons:

          INFO:root:The original frame shape: width=1920, height=1080
          INFO:root:Streaming 29.9fps
          INFO:root:Frame stamp 0.033sec
          INFO:root:Number of frames in video: 74.0
          run_time: 21.19sec
          infer_time: 2.37sec B=2.37, S=0.00
          processing_outputs_time: 0.0861sec
              * av-dur-time: 0.219sec
              * total_counted: 9
    
    - running app on video Pedestrian_Detect_2_1_1.mp4:

          INFO:root:The original frame shape: width=768, height=432
          INFO:root:Streaming 10.0fps
          INFO:root:Frame stamp 0.100sec
          INFO:root:Number of frames in video: 1394.0
          run_time: 72.42sec
          infer_time: 43.36sec B=43.36, S=0.00
          processing_outputs_time: 1.2489sec
    
    - model performance: 
        model ssdlite_mobilenet_v2_coco is very fast but inefficient for person tracking (it's missing a bounding boxes in multiple frames, or detect in one frame several bb for one person) in comparison to the human-pose-estimation-001 from Intel Model Zoo.  The human-pose-estimation-001 performance on the same short video with 2 persons:
  
                INFO:root:The original frame shape: width=1920, height=1080
                INFO:root:Streaming 29.9fps
                INFO:root:Frame stamp 0.033sec
                INFO:root:Number of frames in video: 74.0
                run_time: 188.10sec
                infer_time: 17.05sec B=0.00, S=17.05
                processing_outputs_time: 152.2976sec
                  * av-dur-time: 1.571sec
                  * total_counted: 3
    
- Model 6: [TF MODEL ssd_mobilenet_v2_coco_2018_03_29]
  - [Model source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]

    - CASCADE OF BOUNDING_BOXES MODEL & HUMAN-POSE-ESTIMATION MODEL
  
          - runnin app on short video with 2 persons in each frame:
              models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml
              models/human-pose-estimation-0001.xml
              Device:CPU
              INFO:root:The original frame shape: width=1920, height=1080
              INFO:root:Streaming 30fps
              INFO:root:Frame stamp 0.03sec
              INFO:root:Number of frames in video: 51
              INFO:root:Case of Cascade 2 Models.
              run_time: 133.39sec
              infer_time: 15.20sec b_b=3.59, p=11.61
              processing_outputs_time: 105.9863sec
                * video-dur-time: 1.667sec
                * bb>>av-dur-time: 0.328sec
                * p>> av-dur-time: 1.667sec
                * bb>>total_counted: 6
                * p >>total_counted: 2
          in this case model human-pose-estimation-0001 got a 100% accurate

- Models: human-pose-estimation-0001 and bb model person-detection-retail-0013 from Intel OpenVINO Model Zoo
     
    - Cascade of two models:
  
        - runnin app on short video with 2 people:
    
              INFO:root:The original frame shape: width=1920, height=1080
              INFO:root:Streaming 29.9fps
              INFO:root:Frame stamp 0.033sec
              INFO:root:Number of frames in video: 74.0
              INFO:root:Case of Cascade 2 Models:
              run_time: 194.36sec
              infer_time: 20.37sec b_b=3.43, p=16.94
              processing_outputs_time: 154.4838sec
                  * video-dur-time: 2.440sec
                  * bb>>av-dur-time: 0.602sec
                  * p>> av-dur-time: 1.571sec
                  * bb>>total_counted: 4
                  * p >>total_counted: 3
          in this case the person-detection-retail-0013 performs better than ssd_mobilenet_v2_coco_2018_03_29 but not so accurate as human-pose-estimation-0001 
   
        - running app on video Pedestrian_Detect_2_1_1.mp4:
    
              INFO:root:The original frame shape: width=768, height=432
              INFO:root:Streaming 10.0fps
              INFO:root:Frame stamp 0.100sec
              INFO:root:Number of frames in video: 1394.0
              INFO:root:Case of Cascade 2 Models:
              run_time: 849.06sec
              infer_time: 382.68sec b_b=63.15, p=319.53
              processing_outputs_time: 434.4873sec
                  * video-dur-time: 139.3sec
                  * bb>>av-dur-time: 2.2sec
                  * p>> av-dur-time: 1.85sec
                  * bb>>total_counted: 5
                  * p >>total_counted: 6

The best results I got using IR of model human-pose-estimation-0001 from Intel. 
