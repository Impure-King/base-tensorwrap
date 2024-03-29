import jax
from typing import Any


def is_device_available(device_type: Any = "gpu"):
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
    try:
        jax.devices(device_type.lower())
        return True
    except:
        return False


def list_devices(device_type: str = "gpu"):
    """Returns a list of physical devices that are currently on the device.
    
    Arguments:
        - device_type: The string specifying the type of device to search for. Defaults to gpu.

    Device List:
        - "cpu"
        - "gpu"
        - "tpu"     
    """
    if device_type.lower() == 'cuda':
        device_type = "gpu"
    try:
        devices = jax.devices(device_type.lower())
    except:
        devices = []
    return devices

def set_devices(device_type: str = "gpu"):
    """Sets the global device and completes all the operations on it.
    
    Arguments:
        - device_type: The string specifying the type of device to search for. Defaults to gpu.
        
    Device List:
        - "cpu"
        - "gpu"
        - "tpu" 
    """
    if device_type.lower() == 'cuda':
        device_type = "gpu"
    try:
        devices = jax.config.update("jax_platform_name", device_type.lower())
    except:
        return f"The following device doesn't exist: {device_type}"

def device_put(tensor: jax.Array, device_type:str = "gpu", device_no:int = 0):
    """Transfers an tensor to the specified device and completes all the operations on it.
    
    Arguments:
        - tensor: The tensor to transfer.
        - device_type: The string specifying the type of device to search for. Defaults to gpu.
        - device_no: Specifier for what device to put on. Defaults to 0.
        
    Device List:
        - "cpu"
        - "gpu"
        - "tpu" 
    """

    device = list_devices(device_type)[device_no]
    return jax.device_put(tensor, device)