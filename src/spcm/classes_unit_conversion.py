# -*- coding: utf-8 -*-

import numpy as np

import pint
from . import units

class UnitConversion:
    """
    Conersion functions for units conversion in the package
    """

    @staticmethod
    def convert(value, base_unit, dtype = int, rounding = np.rint):
        if isinstance(value, units.Quantity):
            if rounding is not None:
                return int(rounding(value.to(base_unit).magnitude))
            else:
                return dtype(value.to(base_unit).magnitude)
        return value
    
    @staticmethod
    def to_unit(value, unit : pint.Unit = None):
        if isinstance(value, units.Quantity):
            if isinstance(unit, units.Unit):
                return value.to(unit)
            elif isinstance(unit, units.Quantity):
                return value.to(unit.units)
            else:
                return value.magnitude
        else:
            if isinstance(unit, units.Unit):
                return value * unit
            elif isinstance(unit, units.Quantity):
                return value * unit.units
            else:
                return value
    
    @staticmethod
    def force_unit(value, unit : pint.Unit):
        if isinstance(value, units.Quantity):
            return_value = value.to(unit)
        else:
            return_value =  value * unit
        return return_value