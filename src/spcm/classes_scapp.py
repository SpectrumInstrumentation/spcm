# -*- coding: utf-8 -*-
from .classes_card import Card
from .classes_data_transfer import DataTransfer
from .classes_multi import Multi

from spcm_core import spcm_dwDefTransfer_i64, c_void_p
from spcm_core.constants import *

from .classes_error_exception import SpcmException

_cuda_support = False
try:
    import cupy as cp
    # from cuda import cuda, cudart, nvrtc
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
    from cuda.bindings import nvrtc
except ImportError:
    try:
        from cuda import cuda, cudart, nvrtc
    except ImportError:
        _cuda_support = False
    else:
        _cuda_support = True
else:
    _cuda_support = True

from .classes_unit_conversion import UnitConversion
from . import units


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
    

class SCAPPShared:
    """
    Class for data transfer between the card and the host using the SCAPP API.
    Abstract class that is not meant to be instantiated directly.
    """

    direction : Direction = None
    
    num_threads : int = 1024
    num_blocks : int = 0

    def __init__(self, card : Card, direction : Direction = Direction.Acquisition):
        if not _cuda_support:
            raise ImportError("CUDA support is not available. Please install the cupy and cuda-python packages.")
        super().__init__(card)
        scapp_feature = bool(self.card._features & SPCM_FEAT_SCAPP)
        if not scapp_feature:
            raise SpcmException(text="The card does not have the SCAPP option installed. SCAPP is a add-on feature that needs to be bought separately, please contact info@spec.de and ask for the SCAPP option for the card with serial number {}".format(self.card.sn()))
        self.direction = direction
        self.iterator_index = 0
        self.num_threads = 1024
        self.num_blocks = 0
    
    def _allocate_buffer(self, num_samples : int, no_reshape : bool = False, num_channels : int = 1):
        """
        Memory allocation for the buffer that is used for communicating with the card

        Parameters
        ----------
        num_samples : int | pint.Quantity = None
            use the number of samples an get the number of active channels and bytes per samples directly from the card
        no_reshape : bool = False
            if True, the buffer is not reshaped to the number of channels. This is useful for digital cards where the data is packed in a single array.
        
        Returns
        -------
        numpy array
            the allocated buffer
        """
        
        buffer_size = self.samples_to_bytes(num_samples)
        sample_type = self.numpy_type()

        print(f"{buffer_size = }")
        print(f"{num_samples = }")
        
        # Allocate RDMA buffer
        buffer = cp.empty((buffer_size,), dtype = sample_type, order='F')
        flag = 1
        checkCudaErrors(cuda.cuPointerSetAttribute(flag, cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, buffer.data.ptr))
        if not no_reshape:
            buffer = buffer.reshape((num_channels, -1), order='F')
        return buffer
    
    def start_buffer_transfer(self, *args, buffer_type=SPCM_BUF_DATA, direction=None, notify_samples=None, transfer_offset=None, transfer_length=None, exception_num_samples=False) -> None:
        """
        Setup an RDMA transfer.

        Parameters
        ----------
        *args : list
            Additional commands that are send to the card.
        direction : int
            the direction of the transfer
        notify_samples : int
            Size of the part of the buffer that is used for notifications.
        transfer_offset : int
            the offset of the transfer
        transfer_length : int
            Total length of the transfer buffer.
        """

        self._pre_buffer_transfer(*args, buffer_type=buffer_type, direction=direction, notify_samples=notify_samples, transfer_offset=transfer_offset, transfer_length=transfer_length, exception_num_samples=exception_num_samples)
        if direction is None:
            if self.direction == Direction.Acquisition:
                self._direction = SPCM_DIR_CARDTOGPU
            elif self.direction == Direction.Generation:
                self._direction = SPCM_DIR_GPUTOCARD
            else:
                raise SpcmException(text="Please define a direction for transfer (SPCM_DIR_CARDTOGPU or SPCM_DIR_GPUTOCARD)")

        # we define the buffer for transfer
        self.card._print("Starting the DMA transfer and waiting until data is in board memory")
        self.card._check_error(spcm_dwDefTransfer_i64(self.card._handle, self.buffer_type, self._direction, self.notify_size, c_void_p(self.buffer.data.ptr), self.transfer_offset, self.buffer_size))

        self._post_buffer_transfer(*args, buffer_type=buffer_type, direction=direction, notify_samples=notify_samples, transfer_offset=transfer_offset, transfer_length=transfer_length, exception_num_samples=exception_num_samples)

    
class SCAPPTransfer(DataTransfer, SCAPPShared):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _allocate_buffer(self, num_samples : int, no_reshape : bool = False, num_channels : int = 1):
        return SCAPPShared._allocate_buffer(self, num_samples=num_samples, no_reshape=no_reshape, num_channels=num_channels)
    
    def start_buffer_transfer(self, *args, buffer_type=SPCM_BUF_DATA, direction=None, notify_samples=None, transfer_offset=None, transfer_length=None, exception_num_samples=False) -> None:
        return SCAPPShared.start_buffer_transfer(self, *args, buffer_type=buffer_type, direction=direction, notify_samples=notify_samples, transfer_offset=transfer_offset, transfer_length=transfer_length, exception_num_samples=exception_num_samples)
    
class SCAPPMulti(Multi, SCAPPShared):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _allocate_buffer(self, num_samples : int, no_reshape : bool = False, num_channels : int = 1):
        return SCAPPShared._allocate_buffer(self, num_samples=num_samples, no_reshape=no_reshape, num_channels=num_channels)
    
    def start_buffer_transfer(self, *args, buffer_type=SPCM_BUF_DATA, direction=None, notify_samples=None, transfer_offset=None, transfer_length=None, exception_num_samples=False) -> None:
        return SCAPPShared.start_buffer_transfer(self, *args, buffer_type=buffer_type, direction=direction, notify_samples=notify_samples, transfer_offset=transfer_offset, transfer_length=transfer_length, exception_num_samples=exception_num_samples)
    