#!/usr/bin/env python
# -*- coding: utf-8 -*

"""deepsensemaking (dsm) lang.yaliwc.legacy sub-module"""


import re
import pandas as pd

from collections import OrderedDict

def loadDict(DICT_FILE, items="items", weights="weights", expand=r"[^ \t\n\r\f\v]*", bnd0=r"(?<![a-zA-Z_'\-])(?=[a-zA-Z_'-])", bnd1=r"(?<=[a-zA-Z_'\-])(?![a-zA-Z_'-])", header=0, skiprows=None, ):
    """loadDict

    Load dictionary to be used with YALIWC (i.e., Yet Another Language
    Inquiry and Word Count).  This dictionary should be in CSV format
    and it should contain two columns `items' and `weights'. Items
    column contains strings (words or patterns*) that YALIWC searches
    for in a database of textual data. Weights are values assigned to
    each item (typically ranging from -1 to 1).

    Input args:
    - DICT_FILE,
    - items="items",
    - weights="weights",
    - expand=r"[^ \t\n\r\f\v]*",
    - header=0,
    - skiprows=None,

    Output:
    - dictionary

    Example:
      import os
      import re
      import pandas as pd
      import auxipy
      from auxipy.yaliwc     import loadDict, loadDocs, wordsCounting
      from auxipy.xDictTools import dictPrinter

      from auxipy.yaliwc_dict_dummy_sample import save__yaliwc_dict_dummy_sample
      fn = 'yaliwc_dict_dummy_sample.csv'
      save__yaliwc_dict_dummy_sample(fn)

      dc = loadDict(fn,)
      print("="*77)
      dictPrinter( input_dict=dc, input_name="dc")

    """
    # Read the CSV to dataframe
    df = pd.read_csv( DICT_FILE, header=header, skiprows=skiprows, )
    # Change index to the actual pattern
    df.set_index( items, drop=False, inplace=True, verify_integrity=True, )
    # Make column names usable by YALIWC
    df.rename( columns={ items: "plain", weights: "weight", }, inplace=True, )
    # Convert df to dictionary
    dc = df.to_dict(orient='index',into=OrderedDict)
    # Introduce and compile regular expressions
    # bnd0 = r"\b"
    # bnd1 = r"\b"
    # bnd0 = r"(?=\w)(?<!\w)"
    # bnd1 = r"(?<=\w)(?!\w)"
    # bnd0 = r"(?<![a-zA-Z_'\-])(?=[a-zA-Z_'-])"
    # bnd1 = r"(?<=[a-zA-Z_'\-])(?![a-zA-Z_'-])"
    for key in dc.keys():
        dc[key]["regex"] =  re.compile( bnd0 + re.escape(key).replace(r"\*", expand) + bnd1, re.IGNORECASE, )
    return dc

def loadDocs( DOCS_FILE, docs="tweet_body", header=0, skiprows=None, ):
    df_docs = pd.read_csv( DOCS_FILE, header=header, skiprows=skiprows, )
    assert docs in df_docs.columns, "No %s colum in the documents file" % docs
    df_docs.rename( columns={ docs: "tweet_body",  }, inplace=True, )
    return df_docs

def wordsWeighing(this_row,this_dict,weights="weights",docs="tweet_body",):
    temp_wgt = 0
    for key in this_dict.keys():
        temp_matches = this_dict[key]["regex"].findall(this_row[docs])
        # temp_wgt = temp_wgt + len(temp_matches)*this_dict[key][weights]
        temp_wgt = temp_wgt + len(temp_matches)
    return temp_wgt

def wordsCounting(this_row,this_dict,weights="weights",docs="tweet_body",):
    temp_cnt = 0
    for key in this_dict.keys():
        temp_matches = this_dict[key]["regex"].findall(this_row[docs])
        # temp_cnt = temp_cnt + len(temp_matches)*this_dict[key][weights]
        temp_cnt = temp_cnt + len(temp_matches)
    return temp_cnt

def wordsListingC(this_row,this_dict,weights="weights",docs="tweet_body",):
    # TODO FIXME this one is not so efficient
    # TODO FIXME consider changing of iteration object
    temp_lst = []
    for key in this_dict.keys():
        temp_matches = this_dict[key]["regex"].findall(this_row[docs])
        temp_lst = temp_lst + temp_matches
    return temp_lst

def wordsListingW(this_row,this_dict,weights="weights",docs="tweet_body",):
    """wordsListingW
    return tuples containing words, their counts and their weights
    """
    return None
