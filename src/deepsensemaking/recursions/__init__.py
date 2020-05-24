#!/usr/bin/env python
# -*- coding: utf-8 -*

"""deepsensemaking (dsm) recursions sub-module
see: https://realpython.com/python-thinking-recursively/
"""


import types
import functools
import numpy as np
import re
import datetime as dt

def gen_dict( indict, pre=None, ):
    """
    Example usage:
    =============
    ...
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


def factorial_recursive(n):
    # Base case: 1! = 1
    if n == 1:
        return 1

    # Recursive case: n! = n * (n-1)!
    else:
        return n * factorial_recursive(n-1)


def cumsum(acum_num, curr_iter, max_iter=11, verbose=0,):
    if verbose >= 2:
        print(
            " ".join(
                [
                    "{:>{}d}".format( acum_num, 5 ),
                    "{:>{}d}".format( curr_iter, 5 ),
                    "{:>{}d}".format( max_iter, 5 ),
                ]
            )
        )

    # Base case
    # Return the final state
    if curr_iter == max_iter:
        return acum_num

    # Recursive case
    # Thread the state through the recursive call
    else:
        return cumsum(acum_num + curr_iter, curr_iter + 1,  max_iter=max_iter, verbose=verbose, )
