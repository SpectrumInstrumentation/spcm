import ctypes

from .regs import *
from .spcerr import *

# from .pyspcm import SPCM_DIR_PCTOCARD, SPCM_DIR_CARDTOPC, SPCM_BUF_DATA, SPCM_BUF_ABA, SPCM_BUF_TIMESTAMP

SPCM_DIR_PCTOCARD = 0
SPCM_DIR_CARDTOPC = 1

SPCM_BUF_DATA      = 1000 # main data buffer for acquired or generated samples
SPCM_BUF_ABA       = 2000 # buffer for ABA data, holds the A-DATA (slow samples)
SPCM_BUF_TIMESTAMP = 3000 # buffer for timestamps

TYPE_INT64 = 0
TYPE_DOUBLE = 1

class _U(ctypes.Union):
    """Union for doubles and 64-bit integers"""
    _fields_ = [
        ("dValue", ctypes.c_double),
        ("llValue", ctypes.c_int64)
    ]

class ST_LIST_PARAM(ctypes.Structure):
    """Structure for lists of parameters"""
    _anonymous_ = ("Value",)
    _fields_ = [
        ("lReg", ctypes.c_int32), # the register
        ("lType", ctypes.c_int32), # the type of value written
        ("Value", _U), # the actual value
    ]
