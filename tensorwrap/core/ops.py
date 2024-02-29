from tensorflow import (Tensor,
                        complex,
                        complex64,
                        complex128)
from tensorflow.dtypes import as_dtype

complex_dtypes = [complex,  complex64,  complex128]

def is_complex(input: Tensor) -> bool:
  """Returns True if the data type of input is a complex data type i.e.,
  one of the ``tensorflow.complex``, ``tensorflow.complex64``, and ``tensorflow.complex128``.

  Arguments:
    input (Tensor) - An input tensor.
  """
  return as_dtype(input) in [complex_dtypes]
    