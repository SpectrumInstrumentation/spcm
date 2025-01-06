# -*- coding: utf-8 -*-
import time
import numpy as np
import numpy.typing as npt

from pathlib import Path

from spcm_core import c_void_p, spcm_dwDefTransfer_i64
from spcm_core.constants import *

from .classes_data_transfer import DataTransfer

import pint
from .classes_unit_conversion import UnitConversion
from . import units

from .classes_error_exception import SpcmException, SpcmTimeout


class Gated(DataTransfer):
    """
    A high-level class to control Gated sampling Spectrum Instrumentation cards.

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    Attributes
    ----------
    _pre_trigger : int
        the number of pre trigger samples
    _post_trigger : int
        the number of post trigger samples

    """

    _pre_trigger : int = 0
    _post_trigger : int = 0
    
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
            raise ValueError("The number of post trigger samples needs to be smaller than the total number of samples")
        if num_samples is not None:
            num_samples = UnitConversion.convert(num_samples, units.Sa, int)
            self.card.set_i(SPC_POSTTRIGGER, num_samples)
        post_trigger = self.card.get_i(SPC_POSTTRIGGER)
        return post_trigger
    
    def allocate_buffer(self, num_samples : int, no_reshape = False) -> None:
        """
        Memory allocation for the buffer that is used for communicating with the card

        Parameters
        ----------
        num_samples : int | pint.Quantity = None
            use the number of samples an get the number of active channels and bytes per samples directly from the card
        """
        
        self.buffer_samples = UnitConversion.convert(num_samples, units.Sa, int)

        sample_type = self.numpy_type()

        dwMask = self._buffer_alignment - 1

        item_size = sample_type(0).itemsize
        # allocate a buffer (numpy array) for DMA transfer: a little bigger one to have room for address alignment
        databuffer_unaligned = np.empty(((self._buffer_alignment + self.buffer_size) // item_size, ), dtype = sample_type)   # byte count to sample (// = integer division)
        # two numpy-arrays may share the same memory: skip the begin up to the alignment boundary (ArrayVariable[SKIP_VALUE:])
        # Address of data-memory from numpy-array: ArrayVariable.__array_interface__['data'][0]
        start_pos_samples = ((self._buffer_alignment - (databuffer_unaligned.__array_interface__['data'][0] & dwMask)) // item_size)
        self.buffer = databuffer_unaligned[start_pos_samples:start_pos_samples + (self.buffer_size // item_size)]   # byte address to sample size
        if self.bits_per_sample > 1 and not self._12bit_mode and not no_reshape:
            self.buffer = self.buffer.reshape((self.num_channels, self.buffer_samples), order='F')  # index definition: [channel, sample] !
    
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
            # transfer_offset_bytes = transfer_offset * self.bytes_per_sample * self.num_channels
        else:
            transfer_offset_bytes = 0

        self.buffer_samples = transfer_length
        
        # we define the buffer for transfer and start the DMA transfer
        self.card._print("Starting the DMA transfer and waiting until data is in board memory")
        self._c_buffer = self.buffer.ctypes.data_as(c_void_p)
        self.card._check_error(spcm_dwDefTransfer_i64(self.card._handle, self.buffer_type, direction, self.notify_size, self._c_buffer, transfer_offset_bytes, self.buffer_size))
        
        # Execute additional commands if available
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

    def time_data(self, total_num_samples : int = None) -> npt.NDArray:
        """
        Get the time array for the data buffer

        Parameters
        ----------
        total_num_samples : int | pint.Quantity
            the total number of samples
        
        Returns
        -------
        numpy array
            the time array
        """

        sample_rate = self._sample_rate()
        if total_num_samples is None:
            total_num_samples = self._buffer_samples
        total_num_samples = UnitConversion.convert(total_num_samples, units.Sa, int)
        pre_trigger = UnitConversion.convert(self._pre_trigger, units.Sa, int)
        return ((np.arange(total_num_samples) - pre_trigger) / sample_rate).to_base_units()
    
    def unpack_12bit_buffer(self, data : npt.NDArray[np.int_] = None) -> npt.NDArray[np.int_]:
        """
        Unpack the 12bit buffer to a 16bit buffer

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
    
    def unpackbits(self):
        """
        Unpack the buffer to bits

        Returns
        -------
        numpy array
            the unpacked buffer
        """
        data = self.buffer
        dshape = list(data.shape)
        return_data = data.reshape([-1, 1])
        num_bits = return_data.dtype.itemsize * 8
        mask = 2**np.arange(num_bits, dtype=return_data.dtype).reshape([1, num_bits])
        return (return_data & mask).astype(bool).astype(int).reshape(dshape + [num_bits])

    def tofile(self, filename : str, **kwargs) -> None:
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

        file_path = Path(filename)
        if file_path.suffix == '.bin':
            dtype = kwargs.get('dtype', self.numpy_type())
            self.buffer.tofile(file_path, dtype=dtype)
        elif file_path.suffix == '.csv':
            delimiter = kwargs.get('delimiter', ',')
            np.savetxt(file_path, self.buffer, delimiter=delimiter)
        elif file_path.suffix == '.npy':
            np.save(file_path, self.buffer)
        elif file_path.suffix == '.npz':
            np.savez_compressed(file_path, self.buffer)
        elif file_path.suffix == '.txt':
            np.savetxt(file_path, self.buffer, fmt='%d')
        elif file_path.suffix == '.h5' or file_path.suffix == '.hdf5':
            import h5py
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('data', data=self.buffer)
        else:
            raise ImportError("File format not supported")
        
    def fromfile(self, filename : str, **kwargs) -> None:
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
            self.buffer[:] = buffer.reshape(shape, order='C')
        elif file_path.suffix == '.csv':
            delimiter = kwargs.get('delimiter', ',')
            self.buffer[:] = np.loadtxt(file_path, delimiter=delimiter)
        elif file_path.suffix == '.npy':
            self.buffer[:] = np.load(file_path)
        elif file_path.suffix == '.npz':
            data = np.load(file_path)
            self.buffer[:] = data['arr_0']
        elif file_path.suffix == '.txt':
            self.buffer[:] = np.loadtxt(file_path)
        elif file_path.suffix == '.h5' or file_path.suffix == '.hdf5':
            import h5py
            with h5py.File(file_path, 'r') as f:
                self.buffer[:] = f['data'][()]
        else:
            raise ImportError("File format not supported")

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
        self._pollng_timer = UnitConversion.convert(timer, units.s, float)
    
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

        if self.iterator_index != 0:
            self.avail_card_len(self._notify_samples)

        while True:
            try:
                # print(self.card.status())
                if not self._polling:
                    self.wait_dma()
                else:
                    user_len = self.avail_user_len()
                    if user_len >= self._notify_samples:
                        break
                    time.sleep(0.01)
            except SpcmTimeout:
                self.card._print("... Timeout ({})".format(timeout_counter), end='\r')
                timeout_counter += 1
                if timeout_counter > self._max_timeout:
                    raise StopIteration
            else:
                if not self._polling:
                    break
        
        self.iterator_index += 1

        fill_size = self.fill_size_promille()

        self._current_samples += self._notify_samples
        if self._to_transfer_samples != 0 and self._to_transfer_samples < self._current_samples:
            raise StopIteration

        user_pos = self.avail_user_pos()
        
        # self.card._print("Fill size: {}%  Pos:{:08x} Len:{:08x} Total:{:.2f} MiS / {:.2f} MiS".format(fill_size/10, user_pos, user_len, self._current_samples / MEBI(1), self._to_transfer_samples / MEBI(1)), end='\r', verbose=self._verbose)
        self.card._print("Fill size: {}%  Pos:{:08x} Total:{:.2f} MiS / {:.2f} MiS".format(fill_size/10, user_pos, self._current_samples / MEBI(1), self._to_transfer_samples / MEBI(1)), end='\r', verbose=self._verbose)

        # self.avail_card_len(self._notify_samples) # TODO this probably always a problem! Because the data is not read out yets
        
        return self.buffer[:, user_pos:user_pos+self._notify_samples]