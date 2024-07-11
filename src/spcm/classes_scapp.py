# -*- coding: utf-8 -*-

from .classes_error_exception import SpcmTimeout
from .classes_card import Card
from .classes_data_transfer import DataTransfer
from .pyspcm import spcm_dwDefTransfer_i64, c_void_p
from .constants import SPCM_BUF_DATA, SPCM_DIR_CARDTOGPU, SPCM_DIR_GPUTOCARD, Direction, MEBI

_cuda_support = False
try:
    import cupy as cp
    from cuda import cuda, cudart, nvrtc
except ImportError:
    _cuda_support = False
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
    
    
class SCAPPTransfer(DataTransfer):
    """
    Class for data transfer between the card and the host using the SCAPP API.

    Parameters
    ----------
    direction : Direction = Direction.Acquisition
        Direction of the data transfer.
    """

    direction : Direction = None

    def __init__(self, card : Card, direction : Direction = Direction.Acquisition):
        if not _cuda_support:
            raise ImportError("CUDA support is not available. Please install the cupy and cuda-python packages.")
        super().__init__(card)
        self.direction = direction
        self.iterator_index = 0

    def allocate_buffer(self, num_samples : int) -> None:
        """
        Memory allocation for the buffer that is used for communicating with the card

        Parameters
        ----------
        num_samples : int | pint.Quantity = None
            use the number of samples an get the number of active channels and bytes per samples directly from the card
        """
        
        self.buffer_samples = UnitConversion.convert(num_samples, units.Sa, int)
        # Allocate RDMA buffer
        self.buffer = cp.empty((self.num_channels, self.buffer_samples), dtype = self.numpy_type(), order='F')
        flag = 1
        checkCudaErrors(cuda.cuPointerSetAttribute(flag, cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, self.buffer.data.ptr))
    
    def start_buffer_transfer(self, *args, notify_samples = None, transfer_length = None) -> None:
        """
        Setup an RDMA transfer.

        Parameters
        ----------
        args : int
            Additional commands that are send to the card.
        notify_samples : int
            Size of the part of the buffer that is used for notifications.
        transfer_length : int
            Total length of the transfer buffer.
        """

        self.notify_samples(notify_samples)
        self.buffer_samples = transfer_length

        # Define transfer CUDA buffers
        if self.direction == Direction.Acquisition:
            direction = SPCM_DIR_CARDTOGPU
        else:
            direction = SPCM_DIR_GPUTOCARD
        self.card._check_error(spcm_dwDefTransfer_i64(self.card._handle, SPCM_BUF_DATA, direction, self.notify_size, c_void_p(self.buffer.data.ptr), 0, self.buffer_size))

        # Execute additional commands if available
        if args:
            cmd = 0
            for arg in args:
                cmd |= arg
            self.card.cmd(cmd)
            self.card._print("... CUDA data transfer started")
    
    _auto_avail_card_len = True
    def auto_avail_card_len(self, value : bool = None) -> bool:
        """
        Enable or disable the automatic sending of the number of samples that the card can now use for sample data transfer again

        Parameters
        ----------
        value : bool = None
            True to enable, False to disable and None to get the current status

        Returns
        -------
        bool
            the current status
        """
        if value is not None:
            self._auto_avail_card_len = value
        return self._auto_avail_card_len

    def __next__(self) -> tuple:
        """
        This method is called when the next element is requested from the iterator

        Returns
        -------
        npt.ArrayLike
            the next data block
        
        Raises
        ------
        StopIteration
        """
        timeout_counter = 0

        if self.iterator_index != 0 and self._auto_avail_card_len:
            self.avail_card_len(self._notify_samples)

        while True:
            try:
                self.wait_dma()
            except SpcmTimeout:
                self.card._print("... Timeout ({})".format(timeout_counter), end='\r')
                timeout_counter += 1
                if timeout_counter > self._max_timeout:
                    self.iterator_index = 0
                    raise StopIteration
            else:
                break
        
        self.iterator_index += 1

        self._current_samples += self._notify_samples
        if self._to_transfer_samples != 0 and self._to_transfer_samples < self._current_samples:
            self.iterator_index = 0
            raise StopIteration

        user_pos = self.avail_user_pos()
        fill_size = self.fill_size_promille()
        
        self.card._print("Fill size: {}%  Pos:{:08x} Total:{:.2f} MiS / {:.2f} MiS".format(fill_size/10, user_pos, self._current_samples / MEBI(1), self._to_transfer_samples / MEBI(1)), end='\r', verbose=self._verbose)
        
        return self.buffer[:, user_pos:user_pos+self._notify_samples]

