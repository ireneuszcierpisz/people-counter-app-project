""" Checks if there are unsupported layers in the model, if yes logging.error and exit, if not logging.info """

import sys
import logging
logging.getLogger().setLevel(logging.INFO)


def check_layers(net, device):
    logging.info("Checking if all layers in the model are supported.")        
    not_supported_layers = net.not_supported_layers(device)
    
    # exit if unsupported layers and not CPU as device
    if not_supported_layers and device != "CPU":        
        logging.error("Can not implement CPU extension for {}".format(device))
        sys.exit('Exit! Unsupported layers for {}.'.format(device))
    
    # exit if unsupported layers with CPU even though CPU extension's added
    elif not_supported_layers:
        logging.error('There are not supported layers {} for device: {}'.format(not_supported_layers, device))
        sys.exit('Exit! Unsupported layers for {}.'.format(device))
        
    else:
        logging.info('All layers in this model are supported for {}'.format(device))
        
    return