# -*- coding: utf-8 -*-
import time
import numpy as np
import numpy.typing as npt

from pathlib import Path

from spcm_core import c_void_p, spcm_dwDefTransfer_i64
from spcm_core.constants import *

from .classes_functionality import CardFunctionality

import pint
from .classes_unit_conversion import UnitConversion
from . import units

from .classes_error_exception import SpcmException, SpcmTimeout


class DataTransfer(CardFunctionality):
    """
    A high-level class to control Data Transfer to and from Spectrum Instrumentation cards.

    This class is an iterator class that implements the functions `__iter__` and `__next__`.
    This allows the user to supply the class to a for loop and iterate over the data that 
    is transferred from or to the card. Each iteration will return a numpy array with a data
    block of size `notify_samples`. In case of a digitizer you can read the data from that 
    block and process it. In case of a generator you can write data to the block and it's 
    then transferred.

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    Parameters
    ----------
    `buffer` : NDArray[np.int_]
        numpy object that can be used to write data into the spcm buffer
    `buffer_size`: int
        defines the size of the current buffer shared between the PC and the card
    `buffer_type`: int
        defines the type of data in the buffer that is used for the transfer
    `num_channels`: int
        defines the number of channels that are used for the transfer
    `bytes_per_sample`: int
        defines the number of bytes per sample
    `bits_per_sample`: int
        defines the number of bits per sample
    `direction` : Direction = Direction.Acquisition
        Direction of the data transfer.

    """
    # public
    buffer_size : int = 0
    notify_size : int = 0

    direction : Direction = Direction.Acquisition

    buffer_type : int
    num_channels : int = 0
    bytes_per_sample : int = 0
    bits_per_sample : int = 0

    current_user_pos : int = 0

    _polling = False
    _polling_timer = 0

    # private
    _buffer_samples : int = 0
    _notify_samples : int = 0

    # private
    _memory_size : int = 0
    _c_buffer = None # Internal numpy ctypes buffer object
    _buffer_alignment : int = 4096
    _np_buffer : npt.NDArray[np.int_] # Internal object on which the getter setter logic is working
    _bit_buffer : npt.NDArray[np.int_]
    _8bit_mode : bool = False
    _12bit_mode : bool = False
    _pre_trigger : int = 0

    def __init__(self, card, *args, **kwargs) -> None:
        """
        Initialize the DataTransfer object with a card object and additional arguments

        Parameters
        ----------
        card : Card
            the card object that is used for the data transfer
        *args : list
            list of additional arguments
        **kwargs : dict
            dictionary of additional keyword arguments
        """
        
        self.buffer_size = 0
        self.notify_size = 0
        self.num_channels = 0
        self.bytes_per_sample = 0
        self.bits_per_sample = 0

        self.current_user_pos = 0

        self._buffer_samples = 0
        self._notify_samples = 0
        self._memory_size = 0
        self._c_buffer = None
        self._buffer_alignment = 4096
        self._np_buffer = None
        self._bit_buffer = None
        self._8bit_mode = False
        self._12bit_mode = False
        self._pre_trigger = 0
        
        super().__init__(card, *args, **kwargs)
        self.buffer_type = SPCM_BUF_DATA
        self._bytes_per_sample()
        self._bits_per_sample()
        self.num_channels = self.card.active_channels()

        # Find out the direction of transfer
        if self.function_type == SPCM_TYPE_AI or self.function_type == SPCM_TYPE_DI:
            self.direction = Direction.Acquisition
        elif self.function_type == SPCM_TYPE_AO or self.function_type == SPCM_TYPE_DO:
            self.direction = Direction.Generation
        else:
            self.direction = Direction.Undefined
    

    @property
    def buffer(self) -> npt.NDArray[np.int_]:
        """
        The numpy buffer object that interfaces the Card and can be written and read from
        
        Returns
        -------
        numpy array
            the numpy buffer object with the following array index definition: 
            `[channel, sample]`
            or in case of multiple recording / replay:
            `[segment, sample, channel]`
        """
        return self._np_buffer
    
    @buffer.setter
    def buffer(self, value) -> None:
        self._np_buffer = value
    
    @buffer.deleter
    def buffer(self) -> None:
        del self._np_buffer
    
    @property
    def bit_buffer(self) -> npt.NDArray[np.int_]:
        """
        The bit buffer object that interfaces the Card and can be written and read from
        
        Returns
        -------
        numpy array
            with the buffer object where all the individual bits are now unpacked
        """

        # self._bit_buffer = self.unpackbits(self._np_buffer) # not a good solution
        return self._bit_buffer
    
    @bit_buffer.setter
    def bit_buffer(self, value) -> None:
        self._bit_buffer = value
    
    @bit_buffer.deleter
    def bit_buffer(self) -> None:
        del self._bit_buffer

    @property
    def buffer_samples(self) -> int:
        """
        The number of samples in the buffer
        
        Returns
        -------
        int
            the number of samples in the buffer
        """
        return self._buffer_samples
    
    @buffer_samples.setter
    def buffer_samples(self, value) -> None:
        if value is not None:
            self._buffer_samples = value
            self.buffer_size = self.samples_to_bytes(self._buffer_samples)
    
    @buffer_samples.deleter
    def buffer_samples(self) -> None:
        del self._buffer_samples

    def _bits_per_sample(self) -> int:
        """
        Get the number of bits per sample

        Returns
        -------
        int
            number of bits per sample
        """
        if self._8bit_mode:
            self.bits_per_sample = 8
        elif self._12bit_mode:
            self.bits_per_sample = 12        
        else:
            self.bits_per_sample = self.card.bits_per_sample()
        return self.bits_per_sample
    
    def _bytes_per_sample(self) -> int:
        """
        Get the number of bytes per sample

        Returns
        -------
        int
            number of bytes per sample
        """
        if self._8bit_mode:
            self.bytes_per_sample = 1
        elif self._12bit_mode:
            self.bytes_per_sample = 1.5
        else:
            self.bytes_per_sample = self.card.bytes_per_sample()
        return self.bytes_per_sample
    
    def bytes_to_samples(self, num_bytes : int) -> int:
        """
        Convert bytes to samples
        
        Parameters
        ----------
        bytes : int
            the number of bytes
        
        Returns
        -------
        int
            the number of samples
        """

        if self.bits_per_sample > 1:
            num_samples = num_bytes // self.bytes_per_sample // self.num_channels
        else:
            num_samples = num_bytes // self.num_channels * 8
        return num_samples
    
    def samples_to_bytes(self, num_samples : int) -> int:
        """
        Convert samples to bytes
        
        Parameters
        ----------
        num_samples : int
            the number of samples
        
        Returns
        -------
        int
            the number of bytes
        """

        if self.bits_per_sample > 1:
            num_bytes = num_samples * self.bytes_per_sample * self.num_channels
        else:
            num_bytes = num_samples * self.num_channels // 8
        return num_bytes
    
    def notify_samples(self, notify_samples : int = None) -> int:
        """
        Set the number of samples to notify the user about
        
        Parameters
        ----------
        notify_samples : int | pint.Quantity
            the number of samples to notify the user about
        """

        if notify_samples is not None:
            notify_samples = UnitConversion.convert(notify_samples, units.Sa, int)
            self._notify_samples = notify_samples
            self.notify_size = self.samples_to_bytes(self._notify_samples)
        return self._notify_samples

    def _sample_rate(self) -> pint.Quantity:
        """
        Get the sample rate of the card

        Returns
        -------
        pint.Quantity
            the sample rate of the card in Hz as a pint quantity
        """
        return self.card.get_i(SPC_SAMPLERATE) * units.Hz

    def memory_size(self, memory_samples : int = None) -> int:
        """
        Sets the memory size in samples per channel. The memory size setting must be set before transferring 
        data to the card. (see register `SPC_MEMSIZE` in the manual)
        
        Parameters
        ----------
        memory_samples : int | pint.Quantity
            the size of the memory in samples
        
        Returns
        -------
        int
            the size of the memory in samples
        """

        if memory_samples is not None:
            memory_samples = UnitConversion.convert(memory_samples, units.Sa, int)
            self.card.set_i(SPC_MEMSIZE, memory_samples)
        self._memory_size = self.card.get_i(SPC_MEMSIZE)
        return self._memory_size
    
    def output_buffer_size(self, buffer_samples : int = None) -> int:
        """
        Set the size of the output buffer (see register `SPC_DATA_OUTBUFSIZE` in the manual)
        
        Parameters
        ----------
        buffer_samples : int | pint.Quantity
            the size of the output buffer in Bytes
        
        Returns
        -------
        int
            the size of the output buffer in Samples
        """

        if buffer_samples is not None:
            buffer_samples = UnitConversion.convert(buffer_samples, units.B, int)
            buffer_size = self.samples_to_bytes(buffer_samples)
            self.card.set_i(SPC_DATA_OUTBUFSIZE, buffer_size)
        return self.bytes_to_samples(self.card.get_i(SPC_DATA_OUTBUFSIZE))
    
    def loops(self, loops : int = None) -> int:
        return self.card.loops(loops)
    
    def pre_trigger(self, num_samples : int = None) -> int:
        """
        Set the number of pre trigger samples (see register `SPC_PRETRIGGER` in the manual)
        
        Parameters
        ----------
        num_samples : int | pint.Quantity
            the number of pre trigger samples
        
        Returns
        -------
        int
            the number of pre trigger samples
        """

        if num_samples is not None:
            num_samples = UnitConversion.convert(num_samples, units.Sa, int)
            self.card.set_i(SPC_PRETRIGGER, num_samples)
        self._pre_trigger = self.card.get_i(SPC_PRETRIGGER)
        return self._pre_trigger
    
    def post_trigger(self, num_samples : int = None) -> int:
        """
        Set the number of post trigger samples (see register `SPC_POSTTRIGGER` in the manual)
        
        Parameters
        ----------
        num_samples : int | pint.Quantity
            the number of post trigger samples
        
        Returns
        -------
        int
            the number of post trigger samples
        """

        if self._memory_size < num_samples:
            raise ValueError(f"The number of post trigger samples needs to be smaller than the total number of samples: {self._memory_size} < {num_samples}")
        if num_samples is not None:
            num_samples = UnitConversion.convert(num_samples, units.Sa, int)
            self.card.set_i(SPC_POSTTRIGGER, num_samples)
        post_trigger = self.card.get_i(SPC_POSTTRIGGER)
        self._pre_trigger = self._memory_size - post_trigger
        return post_trigger
    
    def allocate_buffer(self, num_samples : int, no_reshape = False) -> None:
        """
        Memory allocation for the buffer that is used for communicating with the card

        Parameters
        ----------
        num_samples : int | pint.Quantity = None
            use the number of samples an get the number of active channels and bytes per samples directly from the card
        no_reshape : bool = False
            if True, the buffer is not reshaped to the number of channels. This is useful for digital cards where the data is packed in a single array. 
            As well as for 12bit cards where three data points are packed in four bytes or two 16-bit samples.
        """

        self.buffer_samples = UnitConversion.convert(num_samples, units.Sa, int)
        no_reshape |= self._12bit_mode | self.bits_per_sample == 1
        self.buffer = self._allocate_buffer(self.buffer_samples, no_reshape, self.num_channels)
        if self.bits_per_sample == 1:
            self.unpackbits() # allocate the bit buffer for digital cards

    
    def _allocate_buffer(self, num_samples : int, no_reshape : bool = False, num_channels : int = 1) -> npt.NDArray:
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

        dwMask = self._buffer_alignment - 1

        item_size = sample_type(0).itemsize
        # allocate a buffer (numpy array) for DMA transfer: a little bigger one to have room for address alignment
        databuffer_unaligned = np.empty(((self._buffer_alignment + buffer_size) // item_size, ), dtype = sample_type)   # byte count to sample (// = integer division)
        # two numpy-arrays may share the same memory: skip the begin up to the alignment boundary (ArrayVariable[SKIP_VALUE:])
        # Address of data-memory from numpy-array: ArrayVariable.__array_interface__['data'][0]
        start_pos_samples = ((self._buffer_alignment - (databuffer_unaligned.__array_interface__['data'][0] & dwMask)) // item_size)
        buffer = databuffer_unaligned[start_pos_samples:start_pos_samples + (buffer_size // item_size)]   # byte address to sample size
        if not no_reshape:
            buffer = buffer.reshape((num_channels, num_samples), order='F')  # index definition: [channel, sample] !
        return buffer
    
    def start_buffer_transfer(self, *args, buffer_type=SPCM_BUF_DATA, direction=None, notify_samples=None, transfer_offset=None, transfer_length=None, exception_num_samples=False) -> None:
        """
        Start the transfer of the data to or from the card  (see the API function `spcm_dwDefTransfer_i64` in the manual)
        
        Parameters
        ----------
        *args : list
            list of additonal arguments that are added as flags to the start dma command
        buffer_type : int
            the type of buffer that is used for the transfer
        direction : int
            the direction of the transfer
        notify_samples : int
            the number of samples to notify the user about
        transfer_offset : int
            the offset of the transfer
        transfer_length : int
            the length of the transfer
        exception_num_samples : bool
            if True, an exception is raised if the number of samples is not a multiple of the notify samples. The automatic buffer handling only works with the number of samples being a multiple of the notify samples.

        Raises
        ------
        SpcmException
        """

        self.notify_samples(UnitConversion.convert(notify_samples, units.Sa, int))
        transfer_offset = UnitConversion.convert(transfer_offset, units.Sa, int)
        transfer_length = UnitConversion.convert(transfer_length, units.Sa, int)

        if transfer_length is not None:
            self.buffer_samples = transfer_length

        if self.buffer is None: 
            raise SpcmException(text="No buffer defined for transfer")
        if buffer_type: 
            self.buffer_type = buffer_type
        if direction is None:
            if self.direction == Direction.Acquisition:
                direction = SPCM_DIR_CARDTOPC
            elif self.direction == Direction.Generation:
                direction = SPCM_DIR_PCTOCARD
            else:
                raise SpcmException(text="Please define a direction for transfer (SPCM_DIR_CARDTOPC or SPCM_DIR_PCTOCARD)")
        
        if self._notify_samples != 0 and np.remainder(self.buffer_samples, self._notify_samples) and exception_num_samples:
            raise SpcmException("The number of samples needs to be a multiple of the notify samples.")

        if transfer_offset:
            transfer_offset_bytes = self.samples_to_bytes(transfer_offset)
        else:
            transfer_offset_bytes = 0
        
        # we define the buffer for transfer
        self.card._print("Starting the DMA transfer and waiting until data is in board memory")
        self._c_buffer = self.buffer.ctypes.data_as(c_void_p)
        self.card._check_error(spcm_dwDefTransfer_i64(self.card._handle, self.buffer_type, direction, self.notify_size, self._c_buffer, transfer_offset_bytes, self.buffer_size))
        
        # Execute additional commands if available
        if args:
            cmd = 0
            for arg in args:
                cmd |= arg
            self.card.cmd(cmd)
            self.card._print("... data transfer started")

    def duration(self, duration : pint.Quantity, pre_trigger_duration : pint.Quantity = None, post_trigger_duration : pint.Quantity = None) -> None:
        """
        Set the duration of the data transfer
        
        Parameters
        ----------
        duration : pint.Quantity
            the duration of the data transfer
        pre_trigger_duration : pint.Quantity = None
            the duration before the trigger event
        post_trigger_duration : pint.Quantity = None
            the duration after the trigger event
        
        Returns
        -------
        pint.Quantity
            the duration of the data transfer
        """

        if pre_trigger_duration is None and post_trigger_duration is None:
            raise ValueError("Please define either pre_trigger_duration or post_trigger_duration")

        memsize_min = self.card.get_i(SPC_AVAILMEMSIZE_MIN)
        memsize_max = self.card.get_i(SPC_AVAILMEMSIZE_MAX)
        memsize_stp = self.card.get_i(SPC_AVAILMEMSIZE_STEP)
        num_samples = (duration * self._sample_rate()).to_base_units().magnitude
        num_samples = np.ceil(num_samples / memsize_stp) * memsize_stp
        num_samples = np.clip(num_samples, memsize_min, memsize_max)
        num_samples = int(num_samples)
        self.memory_size(num_samples)
        self.allocate_buffer(num_samples)
        if pre_trigger_duration is not None:
            pre_min = self.card.get_i(SPC_AVAILPRETRIGGER_MIN)
            pre_max = self.card.get_i(SPC_AVAILPRETRIGGER_MAX)
            pre_stp = self.card.get_i(SPC_AVAILPRETRIGGER_STEP)
            pre_samples = (pre_trigger_duration * self._sample_rate()).to_base_units().magnitude
            pre_samples = np.ceil(pre_samples / pre_stp) * pre_stp
            pre_samples = np.clip(pre_samples, pre_min, pre_max)
            pre_samples = int(post_samples)
            self.post_trigger(post_samples)
        if post_trigger_duration is not None:
            post_min = self.card.get_i(SPC_AVAILPOSTTRIGGER_MIN)
            post_max = self.card.get_i(SPC_AVAILPOSTTRIGGER_MAX)
            post_stp = self.card.get_i(SPC_AVAILPOSTTRIGGER_STEP)
            post_samples = (post_trigger_duration * self._sample_rate()).to_base_units().magnitude
            post_samples = np.ceil(post_samples / post_stp) * post_stp
            post_samples = np.clip(post_samples, post_min, post_max)
            post_samples = int(post_samples)
            self.post_trigger(post_samples)
        return num_samples, post_samples

    def time_data(self, total_num_samples : int = None, return_units = units.s) -> npt.NDArray:
        """
        Get the time array for the data buffer

        Parameters
        ----------
        total_num_samples : int | pint.Quantity
            the total number of samples
        return_units : pint.Quantity
            the units that the time should be converted to
        
        Returns
        -------
        numpy array
            the time array
        """

        if total_num_samples is None:
            total_num_samples = self._buffer_samples
        total_num_samples = UnitConversion.convert(total_num_samples, units.Sa, int)
        pre_trigger = UnitConversion.convert(self._pre_trigger, units.Sa, int)
        return self.convert_time((np.arange(total_num_samples) - pre_trigger)).to(return_units)
    
    def convert_time(self, time, return_units = units.s):
        """
        Convert a time to the units of the card sample rate
        
        Parameters
        ----------
        time : numpy array
            the time array with integers that should be converted
        return_units : numpy array with pint.Quantity
            the units that the time should be converted to
        
        Returns
        -------
        pint.Quantity
            the converted time
        """

        sample_rate = self._sample_rate()
        return (time / sample_rate).to(return_units)

    def unpack_12bit_buffer(self, data : npt.NDArray[np.int_] = None) -> npt.NDArray[np.int_]:
        """
        Unpacks the 12-bit packed data to 16-bit data
        
        Parameters
        ----------
        data : numpy array
            the packed data

        Returns
        -------
        numpy array
            the unpacked 16bit buffer
        """

        if not self._12bit_mode:
            raise SpcmException("The card is not in 12bit packed mode")
        
        if data is None:
            data = self.buffer

        fst_int8, mid_int8, lst_int8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.int16).T
        nibble_h = (mid_int8 >> 0) & 0x0F
        nibble_m = (fst_int8 >> 4) & 0x0F
        nibble_l = (fst_int8 >> 0) & 0x0F
        fst_int12 = ((nibble_h << 12) >> 4) | (nibble_m << 4) | (nibble_l << 0)
        nibble_h = (lst_int8 >> 4) & 0x0F
        nibble_m = (lst_int8 >> 0) & 0x0F
        nibble_l = (mid_int8 >> 4) & 0x0F
        snd_int12 = ((nibble_h << 12) >> 4) | (nibble_m << 4) | (nibble_l << 0)
        data_int12 = np.concatenate((fst_int12[:, None], snd_int12[:, None]), axis=1).reshape((-1,))
        data_int12 = data_int12.reshape((self.num_channels, self._buffer_samples), order='F')
        return data_int12
    
    def unpackbits(self, data : npt.NDArray[np.int_] = None) -> npt.NDArray[np.int_]:
        """
        Unpack the buffer to bits

        Parameters
        ----------
        data : numpy array | None = None
            the packed data

        Returns
        -------
        numpy array
            the unpacked buffer
        """

        if data is None:
            data = self.buffer
        dshape = list(data.shape)
        return_data = data.reshape([-1, 1])
        num_bits = return_data.dtype.itemsize * 8
        mask = 2**np.arange(num_bits, dtype=return_data.dtype).reshape([1, num_bits])
        self.bit_buffer = (return_data & mask).astype(np.bool).astype(np.uint8).reshape(dshape + [num_bits])
        return self.bit_buffer
    
    def packbits(self) -> None:
        """
        Pack the self.buffer from the self.bit_buffer
        """

        self.buffer[:] = np.packbits(self._bit_buffer, axis=-1, bitorder='little').view(self.buffer.dtype).reshape(self.buffer.shape)

    def tofile(self, filename : str, buffer = None, **kwargs) -> None:
        """
        Export the buffer to a file. The file format is determined by the file extension
        Supported file formats are: 
        * .bin: raw binary file
        * .csv: comma-separated values file
        * .npy: numpy binary file
        * .npz: compressed numpy binary file
        * .txt: whitespace-delimited text file
        * .h5: hdf5 file format

        Parameters
        ----------
        filename : str
            the name of the file that the buffer should be exported to
        
        Raises
        ------
        ImportError
            if the file format is not supported
        """

        if buffer is None:
            buffer = self.buffer
        file_path = Path(filename)
        if file_path.suffix == '.bin':
            buffer.tofile(file_path)
        elif file_path.suffix == '.csv':
            delimiter = kwargs.get('delimiter', ',')
            np.savetxt(file_path, buffer, delimiter=delimiter)
        elif file_path.suffix == '.npy':
            np.save(file_path, buffer)
        elif file_path.suffix == '.npz':
            np.savez_compressed(file_path, buffer)
        elif file_path.suffix == '.txt':
            np.savetxt(file_path, buffer, fmt='%d')
        elif file_path.suffix == '.h5' or file_path.suffix == '.hdf5':
            import h5py
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('data', data=buffer)
        else:
            raise ImportError("File format not supported")
        
    def fromfile(self, filename : str, in_buffer : bool = True, **kwargs) -> npt.NDArray[np.int_]:
        """
        Import the buffer from a file. The file format is determined by the file extension
        Supported file formats are: 
        * .bin: raw binary file
        * .csv: comma-separated values file
        * .npy: numpy binary file
        * .npz: compressed numpy binary file
        * .txt: whitespace-delimited text file
        * .h5: hdf5 file format

        Parameters
        ----------
        filename : str
            the name of the file that the buffer should be imported from
        
        Raises
        ------
        ImportError
            if the file format is not supported
        """

        file_path = Path(filename)
        if file_path.suffix == '.bin':
            dtype = kwargs.get('dtype', self.numpy_type())
            shape = kwargs.get('shape', (self.num_channels, self.buffer_size // self.num_channels))
            buffer = np.fromfile(file_path, dtype=dtype)
            loaded_data = buffer.reshape(shape, order='C')
        elif file_path.suffix == '.csv':
            delimiter = kwargs.get('delimiter', ',')
            loaded_data = np.loadtxt(file_path, delimiter=delimiter)
        elif file_path.suffix == '.npy':
            loaded_data = np.load(file_path)
        elif file_path.suffix == '.npz':
            data = np.load(file_path)
            loaded_data = data['arr_0']
        elif file_path.suffix == '.txt':
            loaded_data = np.loadtxt(file_path)
        elif file_path.suffix == '.h5' or file_path.suffix == '.hdf5':
            import h5py
            with h5py.File(file_path, 'r') as f:
                loaded_data = f['data'][()]
        else:
            raise ImportError("File format not supported")
        
        if in_buffer:
            self.buffer[:] = loaded_data
        return loaded_data


    def avail_card_len(self, available_samples : int = 0) -> None:
        """
        Set the amount of data that has been read out of the data buffer (see register `SPC_DATA_AVAIL_CARD_LEN` in the manual)

        Parameters
        ----------
        available_samples : int | pint.Quantity
            the amount of data that is available for reading
        """

        available_samples = UnitConversion.convert(available_samples, units.Sa, int)
        # print(available_samples, self.bytes_per_sample, self.num_channels)
        available_bytes = self.samples_to_bytes(available_samples)
        self.card.set_i(SPC_DATA_AVAIL_CARD_LEN, available_bytes)
    
    def avail_user_pos(self, in_bytes : bool = False) -> int:
        """
        Get the current position of the pointer in the data buffer (see register `SPC_DATA_AVAIL_USER_POS` in the manual)

        Parameters
        ----------
        in_bytes : bool
            if True, the position is returned in bytes

        Returns
        -------
        int
            pointer position
        """

        self.current_user_pos = self.card.get_i(SPC_DATA_AVAIL_USER_POS)
        if not in_bytes:
            self.current_user_pos = self.bytes_to_samples(self.current_user_pos)
        return self.current_user_pos
    
    def avail_user_len(self, in_bytes : bool = False) -> int:
        """
        Get the current length of the data in the data buffer (see register `SPC_DATA_AVAIL_USER_LEN` in the manual)

        Parameters
        ----------
        in_bytes : bool
            if True, the length is returned in bytes

        Returns
        -------
        int
            data length available
        """

        user_len = self.card.get_i(SPC_DATA_AVAIL_USER_LEN)
        if not in_bytes:
            user_len = self.bytes_to_samples(user_len)
        return user_len
    
    def fill_size_promille(self, return_unit = None) -> int:
        """
        Get the fill size of the data buffer (see register `SPC_FILLSIZEPROMILLE` in the manual)

        Returns
        -------
        int
            fill size
        """

        return_value = self.card.get_i(SPC_FILLSIZEPROMILLE)
        if return_unit is not None: return_value = UnitConversion.to_unit(return_value * units.promille, return_unit)
        return return_value
    
    def wait_dma(self) -> None:
        """
        Wait for the DMA transfer to finish (see register `M2CMD_DATA_WAITDMA` in the manual)
        """
        
        self.card.cmd(M2CMD_DATA_WAITDMA)
    wait = wait_dma

    def numpy_type(self) -> npt.NDArray[np.int_]:
        """
        Get the type of numpy data from number of bytes

        Returns
        -------
        numpy data type
            the type of data that is used by the card
        """

        if self._8bit_mode:
            return np.uint8
        if self._12bit_mode:
            return np.int8
        if self.bits_per_sample == 1:
            if self.num_channels <= 8:
                return np.uint8
            elif self.num_channels <= 16:
                return np.uint16
            elif self.num_channels <= 32:
                return np.uint32
            return np.uint64
        if self.bits_per_sample <= 8:
            return np.int8
        elif self.bits_per_sample <= 16:
            return np.int16
        elif self.bits_per_sample <= 32:
            return np.int32
        return np.int64

    # Data conversion mode
    def data_conversion(self, mode : int = None) -> int:
        """
        Set the data conversion mode (see register `SPC_DATACONVERSION` in the manual)
        
        Parameters
        ----------
        mode : int
            the data conversion mode
        """

        if mode is not None:
            self.card.set_i(SPC_DATACONVERSION, mode)
        mode = self.card.get_i(SPC_DATACONVERSION)
        self._8bit_mode = (mode == SPCM_DC_12BIT_TO_8BIT or mode == SPCM_DC_14BIT_TO_8BIT or mode == SPCM_DC_16BIT_TO_8BIT)
        self._12bit_mode = (mode == SPCM_DC_12BIT_TO_12BITPACKED)
        self._bits_per_sample()
        self._bytes_per_sample()
        return mode
    
    def avail_data_conversion(self) -> int:
        """
        Get the available data conversion modes (see register `SPC_AVAILDATACONVERSION` in the manual)

        Returns
        -------
        int
            the available data conversion modes
        """
        return self.card.get_i(SPC_AVAILDATACONVERSION)
    
    # Iterator methods

    iterator_index = 0
    _max_timeout = 64

    _to_transfer_samples = 0
    _current_samples = 0
    
    _verbose = False

    def verbose(self, verbose : bool = None) -> bool:
        """
        Set or get the verbose mode for the data transfer

        Parameters
        ----------
        verbose : bool = None
            the verbose mode
        """

        if verbose is not None:
            self._verbose = verbose
        return self._verbose

    def to_transfer_samples(self, samples) -> None:
        """
        This method sets the number of samples to transfer

        Parameters
        ----------
        samples : int | pint.Quantity
            the number of samples to transfer
        """

        samples = UnitConversion.convert(samples, units.Sa, int)
        self._to_transfer_samples = samples
    
    def __iter__(self):
        """
        This method is called when the iterator is initialized

        Returns
        -------
        DataIterator
            the iterator itself
        """

        self.iterator_index = 0
        return self
    
    def polling(self, polling : bool = True, timer : float = 0.01) -> None:
        """
        Set the polling mode for the data transfer otherwise wait_dma() is used

        Parameters
        ----------
        polling : bool
            True to enable polling, False to disable polling
        timer : float | pint.Quantity
            the polling timer in seconds
        """

        self._polling = polling
        self._polling_timer = UnitConversion.convert(timer, units.s, float, rounding=None)
    
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

    def __next__(self) -> npt.ArrayLike:
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
        # notify the card that data is available or read, but only after the first block
        if self.iterator_index != 0 and self._auto_avail_card_len:
            self.flush()

        while True:
            try:
                if not self._polling:
                    self.wait_dma()
                else:
                    user_len = self.avail_user_len()
                    if user_len >= self._notify_samples:
                        break
                    time.sleep(self._polling_timer)
            except SpcmTimeout:
                self.card._print("... Timeout ({})".format(timeout_counter), end='\r')
                timeout_counter += 1
                if timeout_counter > self._max_timeout:
                    self.iterator_index = 0
                    raise StopIteration
            else:
                if not self._polling:
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
    
    def flush(self):
        """
        This method is used to tell the card that a notify size of data is freed up after reading (acquisition) or written to (generation)
        """
        self.avail_card_len(self._notify_samples)