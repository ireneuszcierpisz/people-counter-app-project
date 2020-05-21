# Deploy a People Counter App at the Edge project.

## Intel® Edge AI for IoT Developers Nanodegree Program by Udacity.

### What it Does:
The app detects people in a designated area, providing the number of people in the frame, 
average duration of people in frame and total count, using **Intel®** hardware and software tools.

Please find sample results of App performance on the:

      1. single image [here: https://github.com/ireneuszcierpisz/people-counter-app-project/blob/master/images/4show-I_pd-pe_orig.jpg]
      
      2. very short video with two persons
            a) Simultaneous performance of **tf model ssd_mobilenet_v2_coco** and **pose_estimation model** 
               from Intel Model Zoo [here]https://youtu.be/dMz_8uvoTAA
            b) Cascade two models(outputs b.boxes and heatmap) from Intel OpenVino Model Zoo [here]https://youtu.be/eNtujBDE--0
                  
      3. original Udacity video
            Two models from Intel OpenVINO Open Model Zoo: person-detection-retail-0013 outputs bounding boxes 
            and human-pose-estimation-0001 outputs keypoint heatmaps, used simultaneously 
            for counting and tracking people in video. The [video]https://youtu.be/jvBkiwHOY_g shows the performance of both models 
                      
           
**The counter uses the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit.**


**Usage**: Run inference on an input live stream or video 

      [-h] [-mbb MBB]
      [-ms MS] [-mf MF]
      [-i INPUT] [-d D]
      [-c C] [-pt PT]
      [-mode MODE]
      [-l CPU_EXTENSION]

   required arguments:
  
    -mbb MBB              The location of the bb model XML file
    -ms MS                The location of the semantic model XML file
    -mf MF                Model flag 'B' for Bounding Boxes (output shape
                          [1x1xNx7]) , 'P' for Pose Estimation (output blob
                          shape [1x19x32x57]), 'BP' for cascade both models

  optional arguments:
  
    -i INPUT, --input INPUT
                          Live stream 'CAM' or the location of the video file
                          or image
    -d D                  The device name ('GPU', 'MYRIAD') if not 'CPU'
    -c C                  The color of the bounding boxes to draw; RED, GREEN
                          or BLUE
    -pt PT                The confidence/probability threshold to use with the
                          bounding boxes
    -mode MODE            App mode 'write' for video_writer.write() or 'mqtt'
                          for sending data to MQTT server
    -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                          MKLDNN (CPU)-targeted custom layers. Absolute path
                          to a shared library with the kernels impl.

## Requirements
### Hardware

    6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
    OR use of Intel® Neural Compute Stick 2 (NCS2)
    OR Udacity classroom workspace for the related course

### Software

    Intel® Distribution of OpenVINO™ toolkit 2019 R3 release
    Node v6.17.1
    Npm v3.10.10
    CMake
    MQTT Mosca server

## Running on the CPU

When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at:

/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/

Depending on whether you are using Linux or Mac, the filename will be either libcpu_extension_sse4.so or libcpu_extension.dylib, respectively. (The Linux filename may be different if you are using a AVX architecture)
