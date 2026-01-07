# -*- coding: utf-8 -*-

from spcm_core import uint32, int32,\
    create_string_buffer,\
    spcm_dwGetErrorInfo_i32,\
    byref
from spcm_core.constants import *
import inspect
import types

class SpcmError():
    """a container class for handling driver level errors
        
    Examples
    ----------
    ```python
    error = SpcmError(handle=card.handle())
    error = SpcmError(register=0, value=0, text="Some weird error")
    ```

    Parameters
    ---------
    register : int
        the register address that triggered the error
    
    value : int
        the value that was written to the register address
    
    text : str
        the human-readable text associated with the error
    
    """

    register : int = 0
    value : int = 0
    text : str = ""
    _handle = None
    
    def __init__(self, handle = None, register = None, value = None, text = None) -> None:
        """
        Constructs an error object, either by getting the last error from the card specified by the handle
        or using the information coming from the parameters register, value and text

        Parameters
        ----------
        handle : pyspcm.drv_handle (optional)
            a card handle to obtain the last error
        register, value and text : int, int, str (optional)
            parameters to define an error that is not raised by a driver error
        """
        if handle:
            self._handle = handle
            self.get_info()
        if register: self.register = register
        if value: self.value = value
        if text: self.text = text

    def get_info(self) -> int:
        """
        Gets the last error registered by the card and puts it in the object
    
        Class Parameters
        ----------
        self.register
        self.value
        self.text
    
        Returns
        -------
        int
            Error number of the spcm_dwGetErrorInfo_i32 class
        """

        register = uint32(0)
        value = int32(0)
        text = create_string_buffer(ERRORTEXTLEN)
        dwErr = spcm_dwGetErrorInfo_i32(self._handle, byref(register), byref(value), byref(text))
        self.register = register.value
        self.value = value.value
        self.text = text.value.decode('utf-8')
        return dwErr
    
    def __str__(self) -> str:
        """
        Returns a human-readable text of the last error
    
        Class Parameters
        ----------
        self.register
        self.value
        self.text
    
        Returns
        -------
        str
            the human-readable text as saved in self.text.
        """
        
        return str(self.text)

class SpcmException(Exception):
    """a container class for handling driver level errors

    Examples
    ----------
    ```python
    raise SpcmException(handle=card.handle())
    raise SpcmException(register=0, value=0, text="Some weird error")
    ```
    
    Parameters
    ---------
    error : SpcmError
        the error that induced the raising of the exception
    
    """
    error = None

    def __init__(self, error = None, register = None, value = None, text = None) -> None:
        """
        Constructs exception object and an associated error object, either by getting 
        the last error from the card specified by the handle or using the information 
        coming from the parameters register, value and text

        Parameters
        ----------
        handle : drv_handle (optional)
            a card handle to obtain the last error
        register, value and text : int, int, str (optional)
            parameters to define an error that is not raised by a driver error
        """
        call_stack_depth = 0
        caller_stack = inspect.stack()
        for frame_info in caller_stack:
            frame_module = inspect.getmodule(frame_info.frame)
            if frame_module and frame_module.__name__.startswith('spcm.'):
                call_stack_depth += 1
        # Get the caller’s frame (this __init__ and the library function)
        caller_frame = inspect.stack()[call_stack_depth].frame
        # Create a traceback from the caller’s frame
        self.__traceback__ = types.TracebackType(
            tb_frame=caller_frame,
            tb_lasti=caller_frame.f_lasti,
            tb_lineno=caller_frame.f_lineno,
            tb_next=None
        )
        if error: self.error = error
        if register or value or text:
            self.error = SpcmError(register=register, value=value, text=text)
        
    
    def __str__(self) -> str:
        """
        Returns a human-readable text of the last error connected to the exception
    
        Class Parameters
        ----------
        self.error
    
        Returns
        -------
        str
            the human-readable text as return by the error
        """
        
        return str(self.error)

class SpcmTimeout(Exception):
    """a container class for handling specific timeout exceptions"""
    pass

class SpcmDeviceNotFound(SpcmException):
    """a container class for handling specific device not found exceptions"""
    pass