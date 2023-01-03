import functools

API_NAME = 'tensorwrap'

class api_export(object):
    '''Provides ways to export symbols to the TensorWrap API.'''

    def __init__(self, *args, **kwargs):
        
        self._names = args
        self._api_name
tf_export = functools.partial(api_export, api_name=API_NAME)