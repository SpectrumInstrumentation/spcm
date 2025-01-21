# -*- coding: utf-8 -*-

from spcm_core.constants import *

from .classes_error_exception import SpcmException
from .classes_dds import DDS

import ctypes
from enum import IntEnum


class DDSCommandList(DDS):
    """Abstraction of the set_ptr and register `SPC_REGISTER_LIST` for command streaming in the DDS functionality"""

    class WRITE_MODE(IntEnum):
        NO_CHECK = 0
        EXCEPTION_IF_FULL = 1
        WAIT_IF_FULL = 2

    mode : int = WRITE_MODE.NO_CHECK

    command_list : ctypes._Pointer = None
    commands_transfered : int = 0
    current_index : int = 0
    
    _dtm : int = SPCM_DDS_DTM_SINGLE
    _list_size : int = KIBI(16)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.command_list = None
        self.current_index = 0
        
        self._dtm = SPCM_DDS_DTM_SINGLE
        
        self.list_size = self.default_size()

    def data_transfer_mode(self, mode : int) -> None:
        """
        set the data transfer mode of the DDS

        Parameters
        ----------
        mode : int
            the data transfer mode
        """

        self._dtm = mode
        self.card.set_i(SPC_DDS_DATA_TRANSFER_MODE, mode)
        self.card.set_i(SPC_DDS_CMD, SPCM_DDS_CMD_WRITE_TO_CARD)
        self.list_size = self.default_size()
    
    def default_size(self) -> int:
        """
        automatically determine the size of the commands list
        """

        if self._dtm == SPCM_DDS_DTM_SINGLE:
            return self.card.get_i(SPC_DDS_QUEUE_CMD_MAX) // 2
        elif self._dtm == SPCM_DDS_DTM_DMA:
            return KIBI(16)
        raise SpcmException(text="Data transfer mode not supported.")
    
    def allocate(self) -> None:
        """
        allocate memory for the commands list
        """
        if self.command_list is not None:
            del self.command_list
        elems = (ST_LIST_PARAM * (self._list_size + 1))() # +1 for the write to card command at the end
        self.command_list = ctypes.cast(elems, ctypes.POINTER(ST_LIST_PARAM))
        self.current_index = 0

    def load(self, data : dict, exec_mode : int = SPCM_DDS_CMD_EXEC_AT_TRG, repeat : int = 1) -> None:
        """
        preload the command list with data

        Parameters
        ----------
        data : dict
            the data to be preloaded
        mode : int = SPCM_DDS_CMD_EXEC_AT_TRG
            the mode of execution
        repeat : int = 1
            the number of times to repeat the data, if 0 is given the buffer is filled up with the maximal number of blocks that fit in.
        
        TODO make this possible for multiple different keys
        """

        key = list(data.keys())[0]
        value_list = data[key] # For now only take the first key
        size = len(value_list)
        index = 0
        if repeat == 0:
            # repeat the data until the block is full
            repeat = self.list_size // (2*size)
        for _ in range(repeat):
            for value in value_list:
                # Write value
                self.command_list[index].lReg = key
                self.command_list[index].lType = TYPE_DOUBLE
                self.command_list[index].dValue = value
                index += 1
                # Write trigger mode
                self.command_list[index].lReg = SPC_DDS_CMD
                self.command_list[index].lType = TYPE_INT64
                self.command_list[index].llValue = exec_mode
                index += 1
        self.write_to_card()
        self.current_index = index

    def write_to_card(self) -> None:
        """
        write the command list to the card
        """

        self.command_list[self.current_index].lReg = SPC_DDS_CMD
        self.command_list[self.current_index].lType = TYPE_INT64
        self.command_list[self.current_index].llValue = SPCM_DDS_CMD_WRITE_TO_CARD
        self.current_index += 1

    def write(self) -> None:
        """
        send the currently loaded data to the card
        """

        if self.mode == self.WRITE_MODE.EXCEPTION_IF_FULL:
            if self.avail_user_len() < (self.current_index) * ctypes.sizeof(ST_LIST_PARAM):
                raise SpcmException(text="Buffer is full")
        elif self.mode == self.WRITE_MODE.WAIT_IF_FULL:
            timer = 0
            while self.avail_user_len() < (self.current_index) * ctypes.sizeof(ST_LIST_PARAM):
                print("Waiting for buffer to empty {}".format("."*(timer//100)), end="\r")
                timer = (timer + 1) % 400
        self.card.set_ptr(SPC_REGISTER_LIST, self.command_list, (self.current_index) * ctypes.sizeof(ST_LIST_PARAM))
    
    def avail_user_len(self) -> int:
        """
        get the available space for commands in the hardware queue

        TODO: check if this correct. Probably we should use fillsize_promille here
        """

        if self._dtm == SPCM_DDS_DTM_SINGLE:
            return self.list_size - self.card.get_i(SPC_DDS_QUEUE_CMD_COUNT)
        elif self._dtm == SPCM_DDS_DTM_DMA:
            return self.card.get_i(SPC_DATA_AVAIL_USER_LEN)
        else:
            raise SpcmException(text="Data transfer mode not supported.")


    @property
    def list_size(self) -> int:
        """
        get the size of the command list
        """

        return self._list_size

    @list_size.setter
    def list_size(self, size : int) -> None:
        """
        set the size of the command list

        Parameters
        ----------
        size : int
            the size of the command list
        """

        self._list_size = size
        self.allocate()
    
    def reset(self) -> None:
        """
        reset the dds firmware
        """

        # The reset shouldn't be queued!
        self.card.set_i(SPC_DDS_CMD, SPCM_DDS_CMD_RESET)
    