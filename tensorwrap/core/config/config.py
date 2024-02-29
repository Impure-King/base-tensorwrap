import tensorflow as tf
from typing import Any

__all__ = ['is_device_available', 'list_devices', 'set_global_device', 'device_put']

def is_device_available(device_type: str = "gpu") -> bool:
    """Returns a boolean value indicating whether TensorWrap can detect current device. Defaults to cuda detection.
    
    Arguments:
     - device_type: A string indicating what type of device is needed.
    
    Device List:
        - "cpu"
        - "gpu" or "cuda"
        - "tpu"
    """
    
    if device_type == 'cuda':
        device_type = "gpu"
        
    return len(tf.config.list_physical_devices(device_type.upper())) > 0


def list_devices(device_type: str = "gpu"):
    """Returns a list of physical devices that are currently on the device.
    
    Arguments:
        - device_type: The string specifying the type of device to search for. Defaults to gpu.

    Device List:
        - "cpu"
        - "gpu" or "cuda"
        - "tpu"     
    """
    if device_type.lower() == 'cuda':
        device_type = "gpu"
    
    return tf.config.list_physical_devices(device_type.upper())

def set_global_device(device_type: str = "gpu", device_no: int = 0):
    """Sets the global device and completes all the operations on it.
    
    Arguments:
        - device_type: The string specifying the type of device to search for. Defaults to gpu.
        - device_no: The device id, in case of multiple same devices. Defaults to 0.
        
    Device List:
        - "cpu"
        - "gpu" or "cuda"
        - "tpu" 
    """
    
    try:
        tf.config.set_visible_devices(list_devices(device_type)[device_no])
    except:
        return f"The following device doesn't exist: {device_type}"

def device_put(tensor: tf.Tensor, device_type:str = "gpu", device_no:int = 0):
    """Transfers an tensor to the specified device and completes all the operations on it.
    
    Arguments:
        - tensor: The tensor to transfer.
        - device_type: The string specifying the type of device to search for. Defaults to gpu.
        - device_no: Specifier for what device to put on. Defaults to 0.
        
    Device List:
        - "cpu"
        - "gpu" or "cuda"
        - "tpu" 
    """

    if device_type.lower() == "cuda":
        device_type = "gpu"
        
    new_tensor = tensor._copy_to_device(f"{device_type.upper()}:{device_no}")
    return new_tensor