#!/usr/bin/env python
# -*- coding: utf-8 -*

"""deepsensemaking (dsm) dict sub-module"""


import types
import functools
import numpy as np
import re
import datetime as dt

def gen_dict( indict, pre=None, ):
    """
    Example usage:

        di_samp = {}
        di_samp["a1"] = ["A1"]
        di_samp["a2"] = {}
        di_samp["a2"]["b2"] = "A2-B2"
        di_samp["a3"] = {}
        di_samp["a3"]["b3"] = {}
        di_samp["a3"]["b3"]["c3"] = "A3-B3-C3"
        di_samp["a4"] = {}
        di_samp["a4"]["b4"] = {}
        di_samp["a4"]["b4"]["c4"] = {}
        di_samp["a4"]["b4"]["c4"]["d4"] = "A4-B4-C4-D4"

        print( "="*75 )
        print( di_samp )
        print( "-"*75 )

        print( "="*75 )

        from pprint import pprint as pp

        pp( list( gen_dict( indict=di_samp, pre=None, ) ) )

        # See: https://stackoverflow.com/questions/12507206/
    """
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in gen_dict(value, pre + [key]):
                    yield d

            else:
                yield pre + [key, value]

    else:
        yield indict






def dict_str( indict, name="dict"):
    """
    Example usage:

        import pandas as pd
        import re

        di_samp = {}
        di_samp["a1"]  = ["A1",1000]
        di_samp["b1"] = {}
        di_samp["b1"]["b2"] = "B1-B2"
        di_samp["c1"] = {}
        di_samp["c1"]["c2"] = {}
        di_samp["c1"]["c2"]["c3"] = "C1-C2-C3"
        di_samp["d1"] = {}
        di_samp["d1"]["d2"] = {}
        di_samp["d1"]["d2"]["d3"] = {}
        di_samp["d1"]["d2"]["d3"]["d4"] = "D1-D2-D3-D4"
        di_samp["e1"] = 7
        di_samp["f1"] = [1,2,3,4,"5","6","7",8,9,10,11,12,13,14,15]
        di_samp["g1"] = pd.DataFrame({"ok": [ 1, 2, 3, ]})
        di_samp["h1"] = np.array([[1,2,3,4]])
        di_samp["k1"] = re.compile(r"\bwow\b")

        print( "="*75 )
        print( di_samp )
        print( "-"*75 )

        print( "="*75 )

        from

        dictStringer( indict=di_samp, name="di_samp")

    ', '.join(str(x) for x in list_of_ints)

    """

    out_str = ""

    for item in gen_dict( indict ):
        if isinstance(item[-1], ( list, tuple, set ) ):
            if len( item[-1] ) <= 12:
                # print("list:short")
                # print( name + "[\"" + "\"][\"" .join( item[:-1] ) + "\"]" + " = " + "[ " + ", ".join( map( str, item[-1] ) ) + " ]" )
                new_str = name + "[" + "][".join( "\""+str(x)+"\"" if isinstance(x, ( str, ) ) else str(x) for x in item[:-1] ) + "]" + " = " + str(item[-1])
                out_str = out_str + new_str + "\n"
            else:
                # print("list:long")
                # print( name + "[\"" + "\"][\"" .join( item[:-1] ) + "\"]" + " = " + "[ " + ", ".join( map( str, item[-1][0:12] ) ) + "... ]" )
                new_str = name + "[" + "][".join( "\""+str(x)+"\"" if isinstance(x, ( str, ) ) else str(x) for x in item[:-1] ) + "]" + " = " + str(item[-1][0:12]) + " + [ ... ] #sequence was trimmed..."
                out_str = out_str + new_str + "\n"
        elif isinstance(item[-1], ( str, ) ):
            # print("str,")
            new_str = name + "[" + "][".join( "\""+str(x)+"\"" if isinstance(x, ( str, ) ) else str(x) for x in item[:-1] ) + "]" + " = " + "\"" + str(item[-1]) + "\""
            out_str = out_str + new_str + "\n"
        elif isinstance(item[-1], ( re.Pattern, ) ):
            # print("re.Pattern,")
            new_str = name + "[" + "][".join( "\""+str(x)+"\"" if isinstance(x, ( str, ) ) else str(x) for x in item[:-1] ) + "]" + " = " + "\"" + str(item[-1]) + "\""
            out_str = out_str + new_str + "\n"
        elif isinstance(item[-1], ( types.FunctionType, types.BuiltinFunctionType, functools.partial, ) ):
            # print("str,")
            new_str = name + "[" + "][".join( "\""+str(x)+"\"" if isinstance(x, ( str, ) ) else str(x) for x in item[:-1] ) + "]" + " = " + "<" + str( type( item[-1]).__name__ ) + ":" + str(item[-1].__name__) + ">"
            out_str = out_str + new_str + "\n"
        elif isinstance(item[-1], (int, float, complex, ) ):
            # print("int, float, complex,")
            new_str = name + "[" + "][".join( "\""+str(x)+"\"" if isinstance(x, ( str, ) ) else str(x) for x in item[:-1] ) + "]" + " = " + str(item[-1])
            out_str = out_str + new_str + "\n"
        elif isinstance(item[-1], (np.ndarray, np.generic,) ):
            # print("numpy array")
            new_str = name + "[" + "][".join( "\""+str(x)+"\"" if isinstance(x, ( str, ) ) else str(x) for x in item[:-1] ) + "]" + " = <" + str( type( item[-1]).__name__ ) + "> #shape: " + str(item[-1].shape)
            out_str = out_str + new_str + "\n"
        elif isinstance(item[-1], (dt.date,) ):
            new_str = name + "[" + "][".join( "\""+str(x)+"\"" if isinstance(x, ( str, ) ) else str(x) for x in item[:-1] ) + "]" + " = " + item[-1].__repr__() + " #<" + str( type( item[-1]).__name__ ) + ">"
            out_str = out_str + new_str + "\n"




        elif item[-1] is None:
            new_str = name + "[" + "][".join( "\""+str(x)+"\"" if isinstance(x, ( str, ) ) else str(x) for x in item[:-1] ) + "]" + " = " + str(item[-1]) + " #<" + str( type( item[-1]).__name__ ) + ">"
            out_str = out_str + new_str + "\n"
        else:
            # print("other")
            new_str = name + "[" + "][".join( "\""+str(x)+"\"" if isinstance(x, ( str, ) ) else str(x) for x in item[:-1] ) + "]" + " = <" + str( type( item[-1]).__name__ ) + ">"
            out_str = out_str + new_str + "\n"

    return out_str



def dict_print(indict,name="dict"):
    print(dict_str(indict,name=name))
