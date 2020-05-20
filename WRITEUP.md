# Project Write-Up

## Explaining Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

In investigating potential people counter models, I tried each of the following three models:

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
  - running app on single image:
  
    Time of Inference on 1 frame: 0.0403   
    Time of Processing output on 1 frame: 0.001092  
    INFO:root:Size of frame sent to the server: (453, 883, 3)    
    run_time: 0.44sec    
    infer_time: 0.04sec B=0.04, S=0.00
    processing_outputs_time: 0.0011sec
    
  -running app on short video:
    INFO:root:The original frame shape: width=1920, height=1080
    INFO:root:Streaming 29.9fps
    INFO:root:Frame stamp 0.033sec
    INFO:root:Number of frames in video: 74.0
    run_time: 21.19sec
    infer_time: 2.37sec B=2.37, S=0.00
    processing_outputs_time: 0.0861sec
    * av-dur-time: 0.219sec
    * total_counted: 9
  -running app on video Pedestrian_Detect_2_1_1.mp4:
    INFO:root:The original frame shape: width=768, height=432
    INFO:root:Streaming 10.0fps
    INFO:root:Frame stamp 0.100sec
    INFO:root:Number of frames in video: 1394.0
    run_time: 72.42sec
    infer_time: 43.36sec B=43.36, S=0.00
    processing_outputs_time: 1.2489sec
  - model performance: model ssdlite_mobilenet_v2_coco is very fast but inefficient for person tracking, missing a bounding boxes on multiple frames, or detect several bb for one person in comparison to the human-pose-estimation-001 from Intel Model Zoo performance for the same video with 2 persons:
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

  - BOUNDING_BOXES & POSE CASCADE
    - runnin app on video with 2 people:
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
    in this case model human-pose-estimation-0001 100% accurate

- Model human-pose-estimation-0001 and bb model person-detection-retail-0013 from Intel OpenVINO Model Zoo
  - Cascade of two models:
    - runnin app on video with 2 people:
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
   in this case detecting bb model performs better than ssd_mobilenet_v2_coco_2018_03_29 but not so accurate as human-pose-estimation-0001 
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

The best results I got for model human-pose-estimation-0001 IR from Intel. 
