import GPUtil


def list_physical_devices(device="GPU"):
    """Returns a list of physical devices that are currently on the device."""
    return GPUtil.getAvailable()
