#!/usr/bin/env python
# -*- coding: utf-8 -*

"""deepsensemaking (dsm) dict sub-module"""

import types
import functools
import numpy as np
import re
import datetime as dt
import pandas as pd
from functools import reduce
from pprint import pprint as pp
import deepsensemaking as dsm

from dotmap import DotMap

class StuDict(DotMap):
    def __init__(self):
        pass







samp_dict = {}
samp_dict[1] = 1.1
samp_dict["a1"] = "A1"
samp_dict["a2"] = {}
samp_dict["a2"]["b1"] = "A2-B1"
samp_dict["a2"]["b2"] = None
samp_dict["a2"]["b3"] = len
samp_dict["a2"]["b4"] = pd.DataFrame()
samp_dict["a2"]["b5"] = np.eye(3)
samp_dict["a2"]["b6"] = dt.date(2020,1,1)
samp_dict["a2"]["b7"] = "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"
samp_dict["a2"]["b8"] = list(range(5))
# samp_dict["a2"]["b9"] = list(range(120))
samp_dict["a3"] = {}
samp_dict["a3"]["b1"] = range(120)
samp_dict["a3"]["b2"] = set(range(5))
samp_dict["a4"] = {}
samp_dict["a4"]["b1"] = {}
samp_dict["a4"]["b1"]["c1"] = {}
samp_dict["a4"]["b1"]["c1"]["d1"] = "A3-B1-C1-D1"
samp_dict

def reduce_dict(in_dict,max_level=None,):
    """
    Example usage:
    ==============
    from deepsensemaking.dict import samp_dict
    from deepsensemaking.dict import reduce_dict
    print(reduce_dict(in_dict=samp_dict,max_level=1,))

    """
    reducer_seed = tuple()
    def impl(in_dict, pref, level):
        def reducer_func(x, y): return (*x, y)
        def flatten_func(new_in_dict, kv):
            return \
                (max_level is None or level < max_level) \
                and isinstance(kv[1], (dict)) \
                and {**new_in_dict, **impl(kv[1], reducer_func(pref, kv[0]), level + 1)} \
                or {**new_in_dict, reducer_func(pref, kv[0]): kv[1]}

        return reduce(
            flatten_func,
            in_dict.items(),
            {}
        )

    return impl(in_dict, reducer_seed, 0)



def str_dict(in_dict,name="in_dict",max_level=1,disp_vals=True,max_len=40,):
    """
    Example usage:
    ==============
    from deepsensemaking.dict import samp_dict
    from deepsensemaking.dict import str_dict
    print(str_dict(in_dict=samp_dict,max_level=1,,disp_vals=True,max_len=40,))

    """
    repr_func = lambda item: "\""+item+"\"" if isinstance(item, ( str, ) ) else str(item)
    out_str = ""
    for key,val in reduce_dict(in_dict,max_level=max_level,).items():
        out_str += name if name else ""
        out_str += "["
        out_str += "][".join( repr_func(item) for item in key )
        out_str += "]"
        if disp_vals:
            out_str += " = "
            # out_str += str(val)
            val_str = "<???>"
            if isinstance(val,(list,tuple,set,str,int,float,complex,re.Pattern,)):
                val_str = str(val)
                if len(val_str) > max_len:
                    val_str = val_str[:50] + " + [ ... ] # trimmed val..."
            elif val is None:
                val_str = str(val) + " #<" + str(type(val).__name__) + ">"
            elif isinstance(val,(types.FunctionType,types.BuiltinFunctionType,functools.partial,)):
                val_str = "<" + str( type( val).__name__ ) + ":" + str(val.__name__) + ">"
            elif isinstance(val, (np.ndarray, np.generic,) ):
                val_str = "<" + str( type( val).__name__ ) + "> # shape: " + str(val.shape)
            elif isinstance(val, (dt.date,dt.time,dt.datetime,) ):
                val_str = val.__repr__() + " # <" + str( type( val).__name__ ) + ">"
            else:
                val_str = "<" + str( type( val).__name__ ) + ">"
            out_str += val_str

        out_str += "\n"
    return out_str


def print_dict(in_dict,name="in_dict",max_level=1,disp_vals=True,max_len=40,):
    """
    Example usage:
    ==============
    from deepsensemaking.dict import samp_dict
    from deepsensemaking.dict import print_dict
    print_dict(in_dict=samp_dict,max_level=1,disp_vals=True,max_len=40,)

    """
    print(str_dict(in_dict,name=name,max_level=max_level,disp_vals=disp_vals,max_len=max_len,))
