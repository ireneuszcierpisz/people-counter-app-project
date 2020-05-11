import cv2
import logging as log

log.getLogger().setLevel(log.INFO)

""" Checks if the input is image, webcam or video and if allowed open video capture and create the video writer"""

def capture_stream(args):
    image_flag = False # Creates a flag for single images
    
    # Checks if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
        
    # Checks if the input is a single image
    elif args.input.split('.')[-1] in ['jpg', 'gif', 'png', 'tiff', 'bmp']: 
        image_flag = True

    # Gets and open stream/video capture       
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    # Grab the shape of the video frame 
    width = int(cap.get(3))
    height = int(cap.get(4))
    log.info('The original frame shape: width={}, height={}'.format(width, height))

    # # gets Frame Rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_stamp = 1/fps # in sec
    log.info("Streaming {:.1f}fps".format(fps))
    log.info("Frame stamp {:.3f}sec".format(frame_stamp))
    
    # gets Number of Frames in the video file
    if not image_flag:
        nf = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
        log.info("Number of frames in video: {}".format(nf))
    
    # Create a video writer for the output video
    if not image_flag:
        video_writer = cv2.VideoWriter('output_video.mp4', 0x00000021, 30, (width,height))
    else:
        video_writer = None
        
    return cap, video_writer, image_flag, height, width, frame_stamp