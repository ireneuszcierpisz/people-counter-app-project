#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

# import os
# import sys
# import logging as log
# from openvino.inference_engine import IENetwork, IECore


# class Network:
#     """
#     Load and configure inference plugins for the specified target devices 
#     and performs synchronous and asynchronous modes for the specified infer requests.
#     """

#     def __init__(self):
#         ### TODO: Initialize any class variables desired ###

#     def load_model(self):
#         ### TODO: Load the model ###
#         ### TODO: Check for supported layers ###
#         ### TODO: Add any necessary extensions ###
#         ### TODO: Return the loaded inference plugin ###
#         ### Note: You may need to update the function parameters. ###
#         return

#     def get_input_shape(self):
#         ### TODO: Return the shape of the input layer ###
#         return

#     def exec_net(self):
#         ### TODO: Start an asynchronous request ###
#         ### TODO: Return any necessary information ###
#         ### Note: You may need to update the function parameters. ###
#         return

#     def wait(self):
#         ### TODO: Wait for the request to be complete. ###
#         ### TODO: Return any necessary information ###
#         ### Note: You may need to update the function parameters. ###
#         return

#     def get_output(self):
#         ### TODO: Extract and return the output results
#         ### Note: You may need to update the function parameters. ###
#         return

#inference.py

import os
#import sys
import logging
from openvino.inference_engine import IENetwork, IECore
logging.getLogger().setLevel(logging.INFO)

""" Creates class Network with def of function returns not supported layers"""


class Network:
    '''
    Load and store information for working with the Inference Engine and any loaded models.
    '''
    
    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()
        logging.info('Plugin initialized.')
        
        
        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
            logging.info('CPU extension added')
            
            
        # Read the IR as IENetwork
        logging.info('Reading network from IR as IENetwork.')        
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)
        logging.info('IENetwork loaded into the plugin as exec_network.')
        
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        logging.info('Got input_blob from network.imputs generator.')
        
        # Gets the output blobs
        self.output_blob = next(iter(self.network.outputs))
        
        logging.info('Model files:\n{}\n{}\nDevice:{}'.format(model_xml, model_bin, device))
        return

    def list_output_keys(self):
        """
        Gets the network output layer blobs list
        """
        key_list = list(self.network.outputs.keys())
        
        return key_list
    
    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape


    def async_inference(self, image, request_id):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        #logging.info('Starting asynchronous inference')        
        self.exec_network.start_async(request_id=request_id, 
            inputs={self.input_blob: image})
        return


    def wait(self, request_id):
        '''
        Checks the status of the inference request.
        '''
        #logging.info('Waiting for async request to be complete')        
        status = self.exec_network.requests[request_id].wait(-1)
        #logging.info('Status of the inference request: {}'.format(status))        
        return status


    def extract_output(self, request_id, key=None):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        if key:
            output = self.exec_network.requests[request_id].outputs[key]
        else:
            output = self.exec_network.requests[request_id].outputs[self.output_blob]
            
        return output

    
    #Gets not supported layers
    def not_supported_layers(self, device):
            #gets supported layers dictionary
        supported_layers = self.plugin.query_network(self.network, device)
        print('Supported layers:')
        count = 0
        for k, v in supported_layers.items():
            count += 1
            print(' #{:3}   {:40}    {}'.format(count, k, v))
        
        #gets list of not supported layers 
        not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        print('Found: ', len(not_supported_layers),' Unsupported layers:\n', not_supported_layers)

        return not_supported_layers
