import GPUtil


def is_gpu_available():
    """Returns a boolean value indicating whether current system has GPU with cuda or not.
    This however doesn't tell if build is compatible with the current GPU or not."""
    model = GPUtil.getAvailable()
    return True if len(model) >= 1 else False
