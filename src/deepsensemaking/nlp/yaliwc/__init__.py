#!/usr/bin/env python
# -*- coding: utf-8 -*

"""deepsensemaking (dsm) lang.yaliwc sub-module"""


import re
import pandas as pd

from collections import OrderedDict


def dict_item_regex(
        df0,
        wildcard="wildcard",
        weights="weights",
        expand=r"[^ \t\n\r\f\v]*",
        bnd0=r"(?<![a-zA-Z_'\-])(?=[a-zA-Z_'-])",
        bnd1=r"(?<=[a-zA-Z_'\-])(?![a-zA-Z_'-])",
    ):
    """
    import re
    wildcard = "wildcard"
    weights  = "weights"
    expand   = r"[^ \t\n\r\f\v]*"
    bnd0     = r"(?<![a-zA-Z_'\-])(?=[a-zA-Z_'-])"
    bnd1     = r"(?<=[a-zA-Z_'\-])(?![a-zA-Z_'-])"

    from deepsensemaking.lang.yaliwc import dict_item_regex
    this_dict = dict_item_regex(df0_testing_words)
    this_dict

    """
    df0.set_index(wildcard,drop=False,inplace=True,verify_integrity=True,)
    # Introduce and compile regular expressions
    # bnd0 = r"\b"
    # bnd1 = r"\b"
    # bnd0 = r"(?=\w)(?<!\w)"
    # bnd1 = r"(?<=\w)(?!\w)"
    # bnd0 = r"(?<![a-zA-Z_'\-])(?=[a-zA-Z_'-])"
    # bnd1 = r"(?<=[a-zA-Z_'\-])(?![a-zA-Z_'-])"
    df0["regex"] = df0.apply(
        lambda x: re.compile(
            bnd0 + re.escape( x[wildcard] ).replace(r"\*", expand) + bnd1,
            re.IGNORECASE, ),
        axis=1,
    )
    dc0 = df0.to_dict(orient='index',into=OrderedDict,)
    return dc0


def dict_item_weigths(this_row,this_dict,weights="weight",):
    """
    Example usage:
    ==============
    from deepsensemaking.nlp.yaliwc import dict_item_weigths
    from deepsensemaking.nlp.yaliwc import dict_item_counts
    from deepsensemaking.nlp.yaliwc import dict_item_list

    from deepsensemaking.nlp.yaliwc import dict_item_regex

    from deepsensemaking.nlp.yaliwc.dict.testing_words import df0_testing_words
    from deepsensemaking.nlp.yaliwc.dict.testing_words import df1_testing_docs

    this_dict = dict_item_regex(df0_testing_words)
    this_row  = df1_testing_docs.loc[0,"tweet_body"]


    dict_item_weigths(this_row,this_dict,weights="weight",)
    dict_item_counts(this_row,this_dict,weights="weight",)
    dict_item_list(this_row,this_dict,weights="weight",)

    docs = "tweet_body"
    df1_testing_docs["weight"] = df1_testing_docs.apply(lambda x: dict_item_weigths( x[docs],this_dict,weights="weight", ),axis=1,)
    df1_testing_docs["count"]  = df1_testing_docs.apply(lambda x: dict_item_counts(  x[docs],this_dict,weights="weight", ),axis=1,)
    df1_testing_docs["list"]   = df1_testing_docs.apply(lambda x: dict_item_list(    x[docs],this_dict,weights="weight", ),axis=1,)

    df1_testing_docs
    df0_testing_words

    """
    temp_wgt = 0
    for key in this_dict.keys():
        temp_matches = this_dict[key]["regex"].findall(this_row)
        temp_wgt = temp_wgt + len(temp_matches)*this_dict[key][weights]
    return temp_wgt


def dict_item_counts(this_row,this_dict,weights="weights",docs="tweet_body",):
    """
    Example usage:
    ==============
    # See:
    help(dict_item_weigths)

    """
    temp_cnt = 0
    for key in this_dict.keys():
        temp_matches = this_dict[key]["regex"].findall(this_row)
        temp_cnt = temp_cnt + len(temp_matches)
    return temp_cnt


def dict_item_list(this_row,this_dict,weights="weights",docs="tweet_body",):
    """
    Example usage:
    ==============
    # See:
    help(dict_item_weigths)

    """
    # TODO FIXME this one is not so efficient
    # TODO FIXME consider changing of iteration object
    temp_lst = []
    for key in this_dict.keys():
        temp_matches = this_dict[key]["regex"].findall(this_row)
        temp_lst = temp_lst + temp_matches
    return temp_lst


def dict_item_tuple(this_row,this_dict,weights="weights",docs="tweet_body",):
    """dict_item_tuple
    return tuples containing words, their counts and their weights
    """
    return None
