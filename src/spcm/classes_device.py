# -*- coding: utf-8 -*-

import traceback
import types
from typing import List, Dict, Union
import ctypes

from spcm_core import (int64, c_double, c_void_p, create_string_buffer, spcm_hOpen, spcm_vClose, 
                     spcm_dwGetParam_i64, spcm_dwSetParam_i64, 
                     spcm_dwGetParam_d64, spcm_dwSetParam_d64,
                     spcm_dwGetParam_ptr, spcm_dwSetParam_ptr,
                    byref)
from spcm_core.constants import *

from .classes_error_exception import SpcmError, SpcmException, SpcmTimeout, SpcmDeviceNotFound
from .classes_unit_conversion import UnitConversion
from . import units


class Device():
    """
    a class to control the low-level API interface of Spectrum Instrumentation devices

    For more information about what setups are available, please have a look at the user manual
    for your specific device.

    Parameters
    ----------
    device_identifier
        the identifying string that defines the used device

    Raises
    ----------
    SpcmException
    SpcmTimeout
    """

    # public
    device_identifier : str = ""

    # private
    _kwargs : Dict[str, Union[int, float, str]] = {}
    _last_error = None
    _handle = None
    """the handle object used for the card connection"""

    _str_len = 256
    _reraise : bool = False
    _throw_error : bool = True
    _verbose : bool = False
    _closed : bool = True
    """the indicator that indicated whether a connection is opened or closed is set to open (False)"""


    def __init__(self, device_identifier : str = "", handle = False, **kwargs) -> None:
        """Puts the device_identifier in the class parameter self.device_parameter

        Parameters
        ----------
        device_identifier : str = ""
            an identifier string to connect to a specific device, for example:

            * Local PCIe device '/dev/spcm0'
            * Remote 'TCPIP::192.168.1.10::inst0::INSTR'
        handle = False
            directly supply the object with an existing handle
        """
        self.device_identifier = device_identifier
        self._handle = handle
        self._kwargs = kwargs
        self._throw_error = kwargs.get("throw_error", True)
        self._verbose = kwargs.get("verbose", False)
    
    def __del__(self) -> None:
        """Destructor that closes the connection associated with the handle"""
        self.close()

    def __enter__(self) -> object:
        """
        Constructs a handle using the parameter `device_identifier`, when using the with statement
        
        Returns
        -------
        object
            The active card handle
        
        Raises
        ------
        SpcmException
        """
        return self.open()

    def open(self, device_identifier : str = None) -> object:
        """
        Opens a connection to the card and creates a handle, when no with statement is used

        Parameters
        ----------
        device_identifier : str
            The card identifier string (e.g. '/dev/spcm0' for a local device
            NOTE: this is to keep the API consistent with a previous version. The original open()
            method is now in _open()

        Returns
        -------
        object
            This Card object
        """
            
        if device_identifier: # this is to keep the API consistent
            return self.open_handle(device_identifier)
        # This used to be in enter. It is now split up to allow for the open method
        # to be used when no with statement is used
        if self.device_identifier and not self._handle:
            self.open_handle(self.device_identifier)
            if not self._handle and self._throw_error:
                error = SpcmError(text="{} not found...".format(self.device_identifier))
                raise SpcmDeviceNotFound(error)
            # if self._handle:
            #     self._closed = False
        return self
    
    def __exit__(self, exception : SpcmException = None, error_value : str = None, trace : types.TracebackType = None) -> None:
        """
        Handles the exiting of the with statement, when either no code is left or an exception is thrown before

        Parameters
        ----------
        exception : SpcmException
            Only this parameter is used and printed 
        error_value : str
        trace : types.TracebackType

        Raises
        ------
        SpcmException
        """
        if self._verbose and exception:
            self._print("Error type: {}".format(exception))
            self._print("Error value: {}".format(error_value))
            self._print("Traceback:")
            traceback.print_tb(trace)
        elif exception:
            self._print("Error: {}".format(error_value))
        # self.stop(M2CMD_DATA_STOPDMA) # stop the card and the DMA transfer
        self.close()
        if exception and self._reraise:
            raise exception
        
    def close(self) -> None:
        """
        Closes the connection to the card using a handle
        """

        if not self._closed:
            self.stop(M2CMD_DATA_STOPDMA) # stop the card and the DMA transfer
            self.close_handle()
    
    def handle(self) -> object:
        """
        Returns the handle used by the object to connect to the active card
    
        Class Parameters
        ----------
        self._handle
    
        Returns
        -------
        drv_handle
            The active card handle
        """
        
        return self._handle

    # Check if a card was found
    def __bool__(self) -> bool:
        """
        Check for a connection to the active card
    
        Class Parameters
        ----------
        self._handle
    
        Returns
        -------
        bool
            True for an active connection and false otherwise
            
        Examples
        -----------
        >>> card = spcm.Card('/dev/spcm0')
        >>> print(bool(card))
        <<< True # if a card was found at '/dev/spcm0'
        """
        
        return bool(self._handle)
    
    # High-level parameter functions, that use the low-level get and set function    
    def drv_type(self) -> int:
        """
        Get the driver type of the currently used driver (see register `SPC_GETDRVTYPE` in the manual)
    
        Returns
        -------
        int
            The driver type of the currently used driver
        """

        return self.get_i(SPC_GETDRVTYPE)

    def drv_version(self) -> dict:
        """
        Get the version of the currently used driver. (see register `SPC_GETDRVVERSION` in the manual)
    
        Returns
        -------
        dict
            version of the currently used driver
              * "major" - the major version number,
              * "minor" - the minor version number,
              * "build" - the actual build
        """
        version_hex = self.get_i(SPC_GETDRVVERSION)
        major = (version_hex & 0xFF000000) >> 24
        minor = (version_hex & 0x00FF0000) >> 16
        build = version_hex & 0x0000FFFF
        version_dict = {"major": major, "minor": minor, "build": build}
        return version_dict
    
    def kernel_version(self) -> dict:
        """
        Get the version of the currently used kernel. (see register `SPC_GETKERNELVERSION` in the manual)
    
        Returns
        -------
        dict 
            version of the currently used driver
              * "major" - the major version number,
              * "minor" - the minor version number,
              * "build" - the actual build
        """
        version_hex = self.get_i(SPC_GETKERNELVERSION)
        major = (version_hex & 0xFF000000) >> 24
        minor = (version_hex & 0x00FF0000) >> 16
        build = version_hex & 0x0000FFFF
        version_dict = {"major": major, "minor": minor, "build": build}
        return version_dict
    
    def custom_modifications(self) -> dict:
        """
        Get the custom modifications of the currently used device. (see register `SPCM_CUSTOMMOD` in the manual)
    
        Returns
        -------
        dict
            The custom modifications of the currently used device
              * "starhub" - custom modifications to the starhub,
              * "module" -  custom modification of the front-end module(s)
              * "base" - custom modification of the base card
        """

        custom_mode = self.get_i(SPCM_CUSTOMMOD)
        starhub = (custom_mode & SPCM_CUSTOMMOD_STARHUB_MASK) >> 16
        module = (custom_mode & SPCM_CUSTOMMOD_MODULE_MASK) >> 8
        base = custom_mode & SPCM_CUSTOMMOD_BASE_MASK
        custom_dict = {"starhub": starhub, "module": module, "base": base}
        return custom_dict
    
    def log_level(self, log_level : int = None) -> int:
        """
        Set the logging level of the driver
    
        Parameters
        ----------
        log_level : int
            The logging level that is set for the driver
        
        Returns
        -------
        int
            The logging level of the driver
        """

        if log_level is not None:
            self.set_i(SPC_LOGDLLCALLS, log_level)
        return self.get_i(SPC_LOGDLLCALLS)

    def cmd(self, *args) -> None:
        """
        Execute spcm commands (see register `SPC_M2CMD` in the manual)
    
        Parameters
        ----------
        *args : int
            The different command flags to be executed.
        """

        cmd = 0
        for arg in args:
            cmd |= arg
        self.set_i(SPC_M2CMD, cmd)
    
    #@Decorators.unitize(units.ms, "timeout", int)
    def timeout(self, timeout : int = None, return_unit = None) -> int:
        """
        Sets the timeout in ms (see register `SPC_TIMEOUT` in the manual)
        
        Parameters
        ----------
        timeout : int
            The timeout in ms
        
        Returns
        -------
        int
            returns the current timeout in ms
        """

        if timeout is not None:
            timeout = UnitConversion.convert(timeout, units.ms, int)
            self.set_i(SPC_TIMEOUT, timeout)
        return_value = self.get_i(SPC_TIMEOUT)
        return_value = UnitConversion.to_unit(return_value * units.ms, return_unit)
        return return_value
    
    def start(self, *args) -> None:
        """
        Starts the connected card and enables triggering on the card (see command `M2CMD_CARD_START` in the manual)

        Parameters
        ----------
        *args : int
            flags that are send together with the start command
        """

        self.cmd(M2CMD_CARD_START, *args)
    
    def stop(self, *args : int) -> None:
        """
        Stops the connected card (see command `M2CMD_CARD_STOP` in the manual)

        Parameters
        ----------
        *args : int
            flags that are send together with the stop command (e.g. M2CMD_DATA_STOPDMA)
        """

        self.cmd(M2CMD_CARD_STOP, *args)
    
    def reset(self) -> None:
        """
        Resets the connected device (see command `M2CMD_CARD_RESET` in the manual)
        """

        self.cmd(M2CMD_CARD_RESET)

    def write_setup(self, *args) -> None:
        """
        Writes of the configuration registers previously changed to the device (see command `M2CMD_CARD_WRITESETUP` in the manual)

        Parameters
        ----------
        *args : int
            flags that are set with the write command
        """

        self.cmd(M2CMD_CARD_WRITESETUP, *args)
    
    def register_list(self, register_list : List[dict[str, Union[int, float]]]) -> None:
        """
        Writes a list with dictionaries, where each dictionary corresponds to a command (see the user manual of your device for all the available registers)
    
        Parameters
        ----------
        register_list : List[dict[str, Union[int, float]]]
            The list of commands that needs to written to the specific registers of the card.
        """

        c_astParams = (ST_LIST_PARAM * 1024)()
        astParams = ctypes.cast(c_astParams, ctypes.POINTER(ST_LIST_PARAM))
        for i, register in enumerate(register_list):
            astParams[i].lReg = register["lReg"]
            astParams[i].lType = register["lType"]
            if register["lType"] == TYPE_INT64:
                astParams[i].Value.llValue = register["llValue"]
            elif register["lType"] == TYPE_DOUBLE:
                astParams[i].Value.dValue = register["dValue"]
        self.set_ptr(SPC_REGISTER_LIST, astParams, len(register_list) * ctypes.sizeof(ST_LIST_PARAM))

    # Low-level get and set functions
    def get_i(self, register : int) -> int:
        """
        Get the integer value of a specific register of the card (see the user manual of your device for all the available registers)
    
        Parameters
        ----------
        register : int
            The specific register that will be read from.
        
        Returns
        -------
        int
            The value as stored in the specific register
        """

        self._check_closed()
        return_value = int64(0)
        dwErr = spcm_dwGetParam_i64(self._handle, register, byref(return_value))
        self._check_error(dwErr)
        return return_value.value
    get = get_i
    """Alias of get_i"""
    
    def get_d(self, register : int) -> float:
        """
        Get the float value of a specific register of the card (see the user manual of your device for all the available registers)
    
        Parameters
        ----------
        register : int
            The specific register that will be read from.
        
        Returns
        -------
        float
            The value as stored in the specific register
        """

        self._check_closed()
        return_value = c_double(0)
        self._check_error(spcm_dwGetParam_d64(self._handle, register, byref(return_value)))
        return return_value.value
    
    def get_str(self, register : int) -> str:
        """
        Get the string value of a specific register of the card (see the user manual of your device for all the available registers)
    
        Parameters
        ----------
        register : int
            The specific register that will be read from.
        
        Returns
        -------
        str
            The value as stored in the specific register
        """

        self._check_closed()
        return_value = create_string_buffer(self._str_len)
        self._check_error(spcm_dwGetParam_ptr(self._handle, register, byref(return_value), self._str_len))
        return return_value.value.decode('utf-8')
    
    def set_i(self, register : int, value : int) -> None:
        """
        Write the value of a specific register to the card (see the user manual of your device for all the available registers)
    
        Parameters
        ----------
        register : int
            The specific register that will be written.
        value : int
            The value that is written to the card.
        """

        self._check_closed()
        self._check_error(spcm_dwSetParam_i64(self._handle, register, value))
    
    def set_d(self, register : int, value : float) -> None:
        """
        Write the value of a specific register to the card (see the user manual of your device for all the available registers)
    
        Parameters
        ----------
        register : int
            The specific register that will be written.
        value : float
            The value that is written to the card.
        """

        self._check_closed()
        self._check_error(spcm_dwSetParam_d64(self._handle, register, value))
    
    def set_ptr(self, register : int, reference : c_void_p, size : int) -> None:
        """
        Use a memory segment to write to a specific register of the card (see the user manual of your device for all the available registers)
    
        Parameters
        ----------
        register : int
            The specific register that will be read from.
        reference : c_void_p
            pointer to the memory segment
        size : int
            size of the memory segment
        
        Returns
        -------
        int
            The value as stored in the specific register
        """

        self._check_closed()
        self._check_error(spcm_dwSetParam_ptr(self._handle, register, reference, size))

    # Error handling and exception raising
    def _check_error(self, dwErr : int):
        """
        Create an SpcmError object and check for the last error (see the appendix in the user manual of your device for all the possible error codes)
    
        Parameters
        ----------
        dwErr : int
            The error value as returned from a direct driver call
        
        Raises
        ------
        SpcmException
        SpcmTimeout
        """

        # pass
        if dwErr not in [ERR_OK, ERR_TIMEOUT] and self._throw_error:
            self.get_error_info()
            raise SpcmException(self._last_error)
        elif dwErr == ERR_TIMEOUT:
            raise SpcmTimeout("A card timeout occured")

    def get_error_info(self) -> SpcmError:
        """
        Create an SpcmError object and store it in an object parameter
    
        Returns
        ----------
        SpcmError
            the Error object containing the last error
        """

        self._last_error = SpcmError(self._handle)
        return self._last_error
    
    def _check_closed(self) -> None:
        """
        Check if a connection to the card exists and if not throw an error

        Raises
        ------
        SpcmException
        """
        if self._closed:
            error_text = "The connection to the card has been closed. Please reopen the connection before sending commands."
            if self._throw_error:
                raise SpcmException(text=error_text)
            else:
                self._print(error_text)
    
    def _print(self, text : str, verbose : bool = False, **kwargs) -> None:
        """
        Print information

        Parameters
        ----------
        text : str
            The text that is printed
        verbose : bool
            A boolean that indicates if the text should forced to be printed
        **kwargs
            Additional parameters that are passed to the print function
    
        """

        if self._verbose or verbose:
            print(text, **kwargs)

    def open_handle(self, device_identifier : str) -> None:
        """
        Open a connection to the card and create a handle (see the user manual of your specific device on how to find out the device_identifier string)
    
        Parameters
        ----------
        device_identifier : str
            The card identifier string (e.g. '/dev/spcm0' for a local device or 'TCPIP::192.168.1.10::inst0::INSTR' for a remote device)
        """

        self._handle = spcm_hOpen(create_string_buffer(bytes(device_identifier, 'utf-8')))
        self._closed = False
    
    def close_handle(self) -> None:
        """
        Close a connection to the card using the handle
        """

        spcm_vClose(self._handle)
        self._handle = None
        self._closed = True