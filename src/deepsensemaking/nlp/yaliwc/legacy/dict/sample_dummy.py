#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from collections import OrderedDict

dc_yaliwc_dict_dummy_sample = OrderedDict()

dc_yaliwc_dict_dummy_sample[ "self"         ] = 1
dc_yaliwc_dict_dummy_sample[ "self-*"       ] = 1
dc_yaliwc_dict_dummy_sample[ "should*"      ] = 1
dc_yaliwc_dict_dummy_sample[ "skill"        ] = 1
dc_yaliwc_dict_dummy_sample[ "skilled"      ] = 1
dc_yaliwc_dict_dummy_sample[ "skillful*"    ] = 1
dc_yaliwc_dict_dummy_sample[ "skills*"      ] = 1
dc_yaliwc_dict_dummy_sample[ "wit"          ] = 1
dc_yaliwc_dict_dummy_sample[ "wits"         ] = 1
dc_yaliwc_dict_dummy_sample[ "won"          ] = 1
dc_yaliwc_dict_dummy_sample[ "won't"        ] = 1
dc_yaliwc_dict_dummy_sample[ "you"          ] = 1
dc_yaliwc_dict_dummy_sample[ "you'*"        ] = 1
dc_yaliwc_dict_dummy_sample[ "your"         ] = 1


df_yaliwc_dict_dummy_sample = pd.DataFrame.from_dict(
    data    = dc_yaliwc_dict_dummy_sample,
    orient  = 'index',
    columns = ['weights',],
)

df_yaliwc_dict_dummy_sample["items"] = df_yaliwc_dict_dummy_sample.index
df_yaliwc_dict_dummy_sample = df_yaliwc_dict_dummy_sample[["items","weights",]]

def save__yaliwc_dict_dummy_sample( fn="yaliwc_dict_dummy_sample.csv", ):
    df_yaliwc_dict_dummy_sample.to_csv(fn,index=False,)


if __name__ == "__main__":
    save__yaliwc_dict_dummy_sample()
