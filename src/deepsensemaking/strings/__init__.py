#!/usr/bin/env python
# -*- coding: utf-8 -*

"""deepsensemaking (dsm) strings sub-module"""


line0 = "="*77
line1 = "-"*77

def chop_pref(instr,pref,err=True,):
    """chop_pref
    trim chop remove delete prefix

    """
    if len(pref) == 0:
        return instr
    else:
        if instr.startswith(pref):
            return instr[len(pref):]
        elif not err:
            return instr
        else:
            raise ValueError("The {pref} is not a prefix of the input string {instr} hence it cannot be chopped off.".format(instr=repr(instr), pref=repr(pref)))

def chop_suff(instr,suff,err=True,):
    """chop_suff
    trim chop remove delete suffix

    """
    if len(suff) == 0:
        return instr
    else:
        if instr.endswith(suff):
            return instr[:-len(suff)]
        elif not err:
            return instr
        else:
            raise ValueError("The {suff} is not a suffix of the input string {instr} hence it cannot be chopped off.".format(instr=repr(instr), pref=repr(suff)))

def chop_both(instr,pref,suff,err=True,):
    """chop_both
    trim chop remove delete prefix and suffix

    """
    instr = chop_pref( instr, pref, err=err, )
    instr = chop_suff( instr, suff, err=err, )
    return instr
