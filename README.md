# Deploy a People Counter App at the Edge project.

The app detects people in a designated area, providing the number of people in the frame, 
average duration of people in frame, and total count.

The counter uses the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit.


Usage: Run inference on an input live stream or video 

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
