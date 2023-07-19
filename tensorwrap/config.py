import jax


def list_physical_devices(device_type: str = "gpu"):
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

def set_visible_devices(device_type: str = "gpu"):
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
        return "The following device doesn't exist."