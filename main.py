"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from inference import Network
import cv2
import numpy as np
import sys
from get_args import get_args
from capture_stream import capture_stream
from check_layers import check_layers
from preprocess_input import preprocess_frame
from process_output import process_output_bb, process_pose
import logging
import time
import os
import socket
import json
import paho.mqtt.client as mqtt
import ffmpeg

logging.getLogger().setLevel(logging.INFO)

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def connect_mqtt():
    #Connect to the MQTT client
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client


def infer_on_stream(args, start_time, client):
    app_mode = args.mode
    
    device = args.d 
    
    #choose model for inference
    model_flag = args.mf
    if model_flag == "B" or model_flag == "BP":
        net_bb = Network()
        logging.info('IE for model B initialized.')
        model_bb = args.mbb
        # Initialize plugin, add CPU extension and read network from IR
        net_bb.load_model(model_bb, device, CPU_EXTENSION)
        logging.info('Network model_bb loaded into the IE')
        net_bb_output_blobs = net_bb.list_output_keys()
        logging.info('net_bb output blobs: {}'.format(net_bb_output_blobs))
        # checks supported/not supported layers in the model and logging.info or exit
        logging.info('Checking layers in bounding box model..')
        check_layers(net_bb, device)        
    if model_flag == "P" or model_flag == "BP":
        net_s = Network()
        logging.info('IE for model P initialized.')        
        model_s = args.ms
        # Initialize plugin, add CPU extension and read network from IR
        net_s.load_model(model_s, device, CPU_EXTENSION)
        logging.info('Network model_p loaded into the IE.')
        net_s_output_blobs = net_s.list_output_keys()
        logging.info('Output blobs: {}'.format(net_s_output_blobs))        
        # checks supported/not supported layers in the model and logging.info or exit
        logging.info('Checking layers in Pose model..')
        check_layers(net_s, device)     
        
    
    # Get and open video capture
    cap, video_writer, image_flag, height, width = capture_stream(args)
    logging.info('Video captured.')
    
    # creates tracker dict to collect objects location points
    tracker = {} 
    
    persons = [] #list to collect persons/bboxes data
    PDT = {} # dict to collect persons duration times

    if app_mode != "mqtt":
        logging.info("  ! It's NOT  MQTT  MODE! >> for MQTT please use command line argument: -mode 'mqtt'")
    """ Process frames until the video ends, or process is exited """
    logging.info('Processing frames...')
    inferB_time = 0
    inferS_time = 0
    processing_outputB_time = 0
    processing_outputS_time = 0
    
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        
        # Gets Frame Time as ft
        # Gets frame time(ft) as a Current Position of the video file in Milliseconds.
        ft = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Gets 0-based index of the frame to be captured next.
        count = cap.get(cv2.CAP_PROP_POS_FRAMES) 

        if not flag:
            break
        # Waits for a key event for 60 milliseconds
        key_pressed = cv2.waitKey(60)  
        
        # Pre-process the input frame to required shape 
        if model_flag == "B" or model_flag == "BP": 
            frame4infer_bb = preprocess_frame(net_bb, frame)
        if model_flag == "P" or model_flag == "BP":            
            frame4infer_s = preprocess_frame(net_s, frame)
        
        # Perform inference on the frame4infer and get output
        startInfer_time = time.clock()
        request_id = 0
        frame_copy = frame.copy()
        
#BOUNDING BOXES 
        if model_flag == "B" or model_flag == "BP":
            net_bb.async_inference(frame4infer_bb, request_id)                
            if net_bb.wait(request_id) == 0: 
                output_bb = net_bb.extract_output(request_id) 

                inferB_time += time.clock() - startInfer_time #gets inference time
                if image_flag: print("Time of Inference on 1 frame: {:.4}".format(inferB_time))

                # Process the net_bb output
                start_process_output = time.clock() 
                frame_copy, count, tracker, persons, PDT = process_output_bb(output_bb, count, tracker, frame_copy, height, width, args, image_flag, ft, persons, PDT)    
                processing_outputB_time += time.clock() - start_process_output
                if image_flag: print("Time of Processing output on 1 frame: {:.4}".format(processing_outputB_time))
                              
#POSE ESTIMATION
        if model_flag == "P" or model_flag == "BP":
            startInfer_time = time.clock()
            net_s.async_inference(frame4infer_s, request_id)    
            if net_s.wait(request_id) == 0:
                output_s = net_s.extract_output(request_id, "Mconv7_stage2_L2")           

                inferS_time += time.clock() - startInfer_time #gets inference time
                if image_flag: print("Time of Inference on 1 frame: {:.4}".format(inferS_time)) 

                # Process the net_s output
                start_process_output = time.clock()
                frame_copy, count, tracker, persons, PDT, duration, current_count, total_count = process_pose(output_s, count, tracker, frame_copy, height, width, ft, persons, PDT)
                processing_outputS_time += time.clock() - start_process_output
                if image_flag: print("Time of Processing output on 1 frame: {:.4}".format(processing_outputS_time))

                if app_mode == "mqtt":
                    # gets current_count, total_count and duration and send to the MQTT server
                    # Topic "person": keys of "count" and "total" , Topic "person/duration": key of "duration"    
                    client.publish("person", json.dumps({'count': current_count, 'total': total_count}))
                    client.publish("person/duration", json.dumps({'duration': duration}))
                
        if app_mode == "mqtt":
            # Sends frame to the ffmpeg server
            sys.stdout.buffer.write(frame_copy)
            sys.stdout.flush()
                
        # Writes an output image if single_image_mode
        # Write out the frame_copy
        if image_flag:
            cv2.imwrite("output_image.jpg", frame_copy)
            logging.info("        ! Got output image!")
            logging.info('Size of frame sent to the server: {}'.format(frame_copy.shape))                       
        elif app_mode != "mqtt":
            video_writer.write(frame_copy)

                
        # Break if escape key pressed
        if key_pressed == 27:
            break
    
    # Release the video_writer, capture, and destroy any OpenCV windows
    # Close the video writer if the input is not image
    if not image_flag:
        video_writer.release()
        logging.info("          ! Got output_video!")
        
    # Close video file and allow OpenCV to release the captured file
    cap.release()
    # Destroy all of the opened HighGUI windows
    cv2.destroyAllWindows()
    
    if app_mode == 'mqtt':
        # Disconnect from MQTT
        client.disconnect()    
    
    if model_flag == "BP":
        logging.info("Case of Cascade 2 Models:")
    run_time = time.clock() - start_time    
    print("run_time: {:.2f}sec".format(run_time))
    print("infer_time: {:.2f}sec b_b={:.2f}, p={:.2f}".format(inferB_time+inferS_time,inferB_time,inferS_time))
    print("processing_outputs_time: {:.4f}sec".format(processing_outputB_time+processing_outputS_time))

def main():

    start_time = time.clock()

    # Grab command line args
    args = get_args()
    
    app_mode = args.mode
    if app_mode == "mqtt":
        logging.info('MODE=mqtt')    
        # Connect to the MQTT server
        client = connect_mqtt()
    else:
        client = 'client'
    
    # Perform inference on the input stream
    infer_on_stream(args, start_time, client)
    
if __name__ == '__main__':
    main()
