#!/usr/bin/env python
# -*- coding: utf-8 -*

"""deepsensemaking (dsm) dict sub-module"""

import types
import functools
import numpy as np
import datetime as dt
import pandas as pd
from functools import reduce
from pprint import pprint as pp
import deepsensemaking as dsm

import re


try:
    regex_pattern = re._pattern_type
except AttributeError:
    # Python 3.7
    regex_pattern = re.Pattern


samp_dict = {}
samp_dict[1] = 1.1
samp_dict["a1"] = "A1"
samp_dict["a2"] = {}
samp_dict["a2"]["b1"] = "A2-B1"
samp_dict["a2"]["b2"] = None
samp_dict["a2"]["b3"] = len
samp_dict["a2"]["b4"] = lambda x: x
samp_dict["a2"]["b5"] = pd.DataFrame()
samp_dict["a2"]["b6"] = np.eye(3)
samp_dict["a2"]["b7"] = dt.date(2020,1,1)
samp_dict["a2"]["b8"] = "aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeee"
samp_dict["a2"]["b9"] = list(range(5))
# samp_dict["a2"]["b0"] = list(range(120))
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
    from deepsensemaking.dicts import reduce_dict
    from deepsensemaking.dicts import samp_dict
    print(reduce_dict(in_dict=samp_dict,max_level=2,))

    """
    reducer_seed = tuple()
    def impl(in_dict, pref, level):
        def reducer_func(x, y): return (*x, y)
        def flatten_func(in_dict_new, kv):
            return \
                (max_level is None or level < max_level) \
                and isinstance(kv[1], (dict)) \
                and {**in_dict_new, **impl(kv[1], reducer_func(pref, kv[0]), level + 1)} \
                or {**in_dict_new, reducer_func(pref, kv[0]): kv[1]}

        return reduce(
            flatten_func,
            in_dict.items(),
            {}
        )

    return impl(in_dict, reducer_seed, 0)


def str_dict(in_dict,name="in_dict",max_level=None,disp_vals="some",disp_types="some",max_len=42,):
    """
    Example usage:
    ==============
    from deepsensemaking.dicts import str_dict
    from deepsensemaking.dicts import samp_dict
    print(str_dict(in_dict=samp_dict,))

    """
    repr_func = lambda item: "\""+item+"\"" if isinstance(item, ( str, ) ) else str(item)
    out_str = ""
    for key,val in reduce_dict(in_dict,max_level=max_level,).items():
        out_str += name if name else ""
        out_str += "["
        out_str += "][".join( repr_func(item) for item in key )
        out_str += "]"
        val_str = ""
        type_str = ""
        if disp_vals == "some" or disp_vals == "all":
            if isinstance(val,(str,)):
                val_str = " = " + repr_func(val)
            elif isinstance(val,(regex_pattern,)):
                val_str = " = " + repr_func(val)
            elif isinstance(val,(list,tuple,set,)):
                val_str = " = " + repr_func(val).replace(" ", "")
            elif isinstance(val,(int,float,complex,)):
                val_str = " = " + repr_func(val)
            elif val is None:
                val_str = " = " + str(val)
            elif isinstance(val,(types.FunctionType,types.BuiltinFunctionType,functools.partial,)):
                val_str = " # " + val.__repr__()
            elif isinstance(val, (np.ndarray, np.generic,) ):
                val_str = " # " + val.__repr__().replace(" ", "")
            elif isinstance(val, (pd.DataFrame,) ):
                val_str = " # DF"
                if disp_vals == "all":
                    val_str = " # DF\n" + val.__repr__()
            elif isinstance(val, (dt.date,dt.time,dt.datetime,) ):
                val_str = " # " + val.__repr__().replace(" ", "")
            else:
                val_str = " # " + repr_func(val)
            # REPLACEMENTS
            if not isinstance(val,(pd.DataFrame,)):
                val_str = val_str.replace("\n", "").replace("\r", "")
            # TRIMMING
            if len(val_str) > max_len:
                val_str = val_str[:max_len] + " # [...]"
            # ADD SHAPE
            if isinstance(val, (np.ndarray,np.generic,pd.DataFrame) ):
                val_str += " # shape: " + str(val.shape)
            # ADD LEN
            if isinstance(val, (list,tuple,set,) ):
                val_str += " # len: " + str(len(val))
        if disp_types == "some":
            if not isinstance(val,(types.FunctionType,types.BuiltinFunctionType,functools.partial,list,tuple,set,str,int,float,)):
                type_str = " # <" + str(type(val).__name__) + ">"
        elif disp_types == "all":
            if not isinstance(val,(types.FunctionType,types.BuiltinFunctionType,functools.partial,)):
                type_str += " # <" + str(type(val).__name__) + ">"

        out_str += str(val_str)
        out_str += str(type_str)
        out_str += "\n"
    return out_str


def print_dict(in_dict,name="in_dict",max_level=None,disp_vals="some",disp_types="some",max_len=42,):
    """
    Example usage:
    ==============
    from deepsensemaking.dicts import print_dict
    from deepsensemaking.dicts import samp_dict
    print_dict(in_dict=samp_dict,)

    """
    print(
        str_dict(
            in_dict,
            name       = name,
            max_level  = max_level,
            disp_vals  = disp_vals,
            disp_types = disp_types,
            max_len    = max_len,
        )
    )
