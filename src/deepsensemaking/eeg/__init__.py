#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
deepsensemaking (dsm) eeg auxiliary tools


"""

from deepsensemaking.eeg import mne



from deepsensemaking.dicts import str_dict,print_dict
from deepsensemaking.bids  import get_bids_prop


from collections import OrderedDict
import pandas as pd
import numpy as np
import json
import sys

from tqdm.autonotebook import tqdm
tqdm.pandas()


def convert_ZZ_translators(ifSetup="peaks/data/setup.json",verbose=3,):
    # Load study setup
    with open(ifSetup) as fhSetup: setup = OrderedDict(json.load(fhSetup))

    # Projects swapping keys from `str` to `int`
    cond_swaps = OrderedDict()
    for key0,val0 in setup["db_infos_zz"]["cond"].items():
            cond_swaps[int(key0)] = val0

    tmin_swaps = OrderedDict()
    for key0,val0 in setup["db_infos_zz"]["time"].items():
            tmin_swaps[int(key0)] = val0[0]

    tmax_swaps = OrderedDict()
    for key0,val0 in setup["db_infos_zz"]["time"].items():
            tmax_swaps[int(key0)] = val0[1]

    if verbose > 2:
        print_dict(cond_swaps,"cond_swaps")
        print_dict(tmin_swaps,"tmin_swaps")
        print_dict(tmax_swaps,"tmax_swaps")


    # Provide dictionary to translate from channels to channel bundles
    chan_swaps = OrderedDict()
    for key0,val0 in setup["chans"]["bund0"].items():
        for item0 in val0:
            chan_swaps[item0] = key0

    if verbose > 2:
        print_dict(chan_swaps,"chan_swaps")

    return cond_swaps,tmin_swaps,tmax_swaps,chan_swaps





def convert_ZZ(dfZZ,ifSetup="peaks/data/setup.json",verbose=3,):
    """Convert Brain Vision Analizer PEAK export from ZZ to something
    more like my pandas exports

    """

    # Add relevant columns
    cols0 = [
        "evoked0",
        "quest0",
        "cond0",
        "chan0",
        "tmin0",
        "tmax0",
        "mode0",
        "chanX",
        "latX",
        "valX",
        "SUB",
        "SES",
        "TASK",
        "RUN",
        "CHAN_BUND",
    ]
    for col0 in cols0:
        dfZZ[col0] = None


    cond_swaps,tmin_swaps,tmax_swaps,chan_swaps = convert_ZZ_translators(
        ifSetup=ifSetup,verbose=verbose,
    )


    # TRANSLATE
    dfZZ["evoked0"]   = "evokedZ" # This column typically indicates preprocessing stage
    dfZZ["quest0"]    = "word_set" # Here we have only one quest (additionally `word_len` is present in MNE data)
    dfZZ["cond0"]     = dfZZ["COND_NUM"].map(cond_swaps) # POS/m/f derrived conditions
    # dfZZ["chan0"]     = dfZZ["ELEC"].map(chan_swaps) # Translation of channel names to channel bundles
    dfZZ["chan0"]     = dfZZ["ELEC"]
    dfZZ["tmin0"]     = dfZZ["TIME_WINDOW"].map(tmin_swaps)
    dfZZ["tmax0"]     = dfZZ["TIME_WINDOW"].map(tmax_swaps)
    dfZZ["mode0"]     = "pos" # Not so important now but `pos` turns out to be the right option
    # dfZZ["chanX"]     = dfZZ["ELEC"].map(chan_swaps) # same as above
    dfZZ["chanX"]     = dfZZ["ELEC"]
    dfZZ["latX"]      = np.nan # No peak latency data was provided
    dfZZ["valX"]      = dfZZ["PEAK"] # Peak aplitude
    dfZZ["SUB"]       = dfZZ["File"].apply(lambda x: get_bids_prop(x, prop="sub"))  # Extract subject code from file name
    dfZZ["SES"]       = "eeg001" # Session code
    dfZZ["TASK"]      = dfZZ["File"].apply(lambda x: get_bids_prop(x, prop="task")) # Extract task code from file name
    dfZZ["RUN"]       = dfZZ["File"].apply(lambda x: get_bids_prop(x, prop="run"))  # Extract run number from file name

    dfZZ["CHAN_BUND"] = np.nan # This is NaN for bundles that are set in `chan0` and `chanX` columns
    dfZZ["CHAN_BUND"] = dfZZ["ELEC"].map(chan_swaps)


    return dfZZ




def chan_bund_INEFFICIENT_mean(dfPK):
    """
    Very INEFFICIENT way to get mean for every combination of factor
    (variable) levels (values).

    In general it is better to use pd.DataFrame.groupby().
    This approach should be considered to be
    a quick (although quite relaible) hack...

    """

    all_quest0    = dfPK["quest0"]   .unique()
    all_cond0     = dfPK["cond0"]    .unique()
    all_tmin0     = dfPK["tmin0"]    .unique()
    all_mode0     = dfPK["mode0"]    .unique()
    all_SUB       = dfPK["SUB"]      .unique()
    all_SES       = dfPK["SES"]      .unique()
    all_RUN       = dfPK["RUN"]      .unique()
    all_CHAN_BUND = dfPK["CHAN_BUND"].unique()


    """
    from functools import reduce

    all_list = [
        all_quest0,
        all_cond0,
        all_tmin0,
        all_mode0,
        all_SUB,
        all_SES,
        all_RUN,
        all_CHAN_BUND,
    ]

    all_len   = list(map(lambda x: len(x), all_list))
    all_count = reduce((lambda x, y: x * y), all_len)

    """


    all_count = np.prod([
        len( all_quest0    ),
        len( all_cond0     ),
        len( all_tmin0     ),
        len( all_mode0     ),
        len( all_SUB       ),
        len( all_SES       ),
        len( all_RUN       ),
        len( all_CHAN_BUND ),
    ])


    """
    quest0i = all_quest0    [0]
    cond0i  = all_cond0     [0]
    tmin0i  = all_tmin0     [0]
    mode0i  = all_mode0     [0]
    SUBi    = all_SUB       [0]
    SESi    = all_SES       [0]
    RUNi    = all_RUN       [0]
    bund0i  = all_CHAN_BUND [0]

    """

    df_temp0 = pd.DataFrame([],columns=dfPK.columns)

    ii = 0
    with tqdm( total=all_count, ) as pbar:
        for                             quest0i in all_quest0   :
            for                         cond0i  in all_cond0    :
                for                     tmin0i  in all_tmin0    :
                    for                 mode0i  in all_mode0    :
                        for             SUBi    in all_SUB      :
                            for         SESi    in all_SES      :
                                for     RUNi    in all_RUN      :
                                    for bund0i  in all_CHAN_BUND:
                                        pbar.set_description(
                                            "{} {} {} {} {} {} {} {}".format(
                                                quest0i,cond0i,tmin0i,mode0i,SUBi,SESi,RUNi,bund0i,
                                            )
                                        )
                                        pbar.update(1)
                                        ii += 1
                                        df_temp1 = dfPK.query(""" quest0==@quest0i & cond0==@cond0i & tmin0==@tmin0i & mode0==@mode0i & SUB==@SUBi & SES==@SESi & RUN==@RUNi & CHAN_BUND==@bund0i """)
                                        df_temp2 = df_temp1.iloc[:1].copy()
                                        df_temp2["chan0"]     = bund0i
                                        df_temp2["chanX"]     = bund0i
                                        df_temp2["CHAN_BUND"] = np.nan
                                        df_temp2["latX"]      = df_temp1["latX"].mean()
                                        df_temp2["valX"]      = df_temp1["valX"].mean()
                                        df_temp0 = df_temp0.append(df_temp2)


    return df_temp0
