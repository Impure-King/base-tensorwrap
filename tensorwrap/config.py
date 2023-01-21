import GPUtil
from typing import Any


def list_physical_devices(device_type: Any = "GPU"):
    """Returns a list of physical devices that are currently on the device.
       Uses custom indexing when defining the GPU number."""
    devices = GPUtil.getAvailable()
    device_list = []
    for i in range(len(devices)):
        device = f"PhysicalDevice(name='/physical_device:GPU:{i}', device_type='GPU')"
        device_list.append(device)
    return device_list
