#!/usr/bin/env python
# -*- coding: utf-8 -*

"""deepsensemaking (dsm) dict sub-module"""


import re



ex_fname = "sub-123abc_ses-03func_task-01easy02led03corr_run-02_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"

def get_bids_prop( if_name, prop ):
    """
    Example usage:
    ==============
    from deepsensemaking.bids import get_bids_prop
    from deepsensemaking.bids import ex_fname
    print(get_bids_prop(if_name=ex_fname,prop="sub",))

    """
    # r = re.compile( prop + r"-(.*?)(?:_+)" )
    # r = re.compile( prop + r"-(.*?)(?:$)" )
    # r = re.compile( prop + r"-(.*?)(?:_+|$)" )
    r = re.compile( prop + r"-(.*?)(?:_|$)" )
    m = r.findall( if_name )
    if m:
        return m[0]
    else:
        return None
