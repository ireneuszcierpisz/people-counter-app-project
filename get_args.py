#get_args.py

import argparse

MODEL1 = "models/person-detection-retail-0013.xml"
MODEL2 = "models/semantic-segmentation-adas-0001.xml"
MODEL3 = "models/human-pose-estimation-0001.xml"
VIDEO = "images/myvideo_kart.mp4"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input live stream or video")
    
    # Descriptions for the commands
    mbb_desc = "The location of the bb model XML file"
    ms_desc = "The location of the semantic model XML file"
    i_desc = "Live stream 'CAM' or the location of the video file or image"
    d_desc = "The device name ('GPU', 'MYRIAD') if not 'CPU'"
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"
    mf_desc = "Model flag 'B' for Bounding Boxes (output shape [1x1xNx7]) , 'P' for Pose Estimation (output blob shape  [1x19x32x57]), 'BP' for cascade two models"
    
    # Required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    # Arguments
    required.add_argument("-mbb", help=mbb_desc, default= MODEL1)
    required.add_argument("-ms", help=ms_desc, default= MODEL3)
    required.add_argument("-mf", help=mf_desc, default='P')
    optional.add_argument("-i", "--input", help=i_desc, default= VIDEO)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='RED')
    optional.add_argument("-ct", help=ct_desc, type=float, default=0.6)
    args = parser.parse_args()

    return args