import cv2
import numpy as np

# Pre-process the input frame to required shape 
def preprocess_frame(net, frame):
    
    # gets the model input shape
    model_shape = net.get_input_shape()
    model_w = model_shape[3]
    model_h = model_shape[2]
    
    # copying the frame as numpy.ndarray and assignes the returning copy to the frame4infer variable.
    frame4infer = np.copy(frame)
    frame4infer = cv2.resize(frame4infer, (model_w, model_h))
    frame4infer = frame4infer.transpose((2,0,1))
    frame4infer = frame4infer.reshape(1, 3, model_h, model_w)
    
    return frame4infer