#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from collections import OrderedDict

dc_yaliwc_dict_emoji_sample = OrderedDict()

dc_yaliwc_dict_emoji_sample[ ":*"  ] = 1
dc_yaliwc_dict_emoji_sample[ "*:"  ] = 1
dc_yaliwc_dict_emoji_sample[ ":)"  ] = 1
dc_yaliwc_dict_emoji_sample[ "(:"  ] = 1
dc_yaliwc_dict_emoji_sample[ ";)"  ] = 1
dc_yaliwc_dict_emoji_sample[ "(;"  ] = 1
dc_yaliwc_dict_emoji_sample[ ":("  ] = -1
dc_yaliwc_dict_emoji_sample[ "):"  ] = -1
dc_yaliwc_dict_emoji_sample[ ":/"  ] = -1
dc_yaliwc_dict_emoji_sample[ "/:"  ] = -1

df_yaliwc_dict_emoji_sample = pd.DataFrame.from_dict(
    data    = dc_yaliwc_dict_emoji_sample,
    orient  = 'index',
    columns = ['weights',],
)

df_yaliwc_dict_emoji_sample["items"] = df_yaliwc_dict_emoji_sample.index
df_yaliwc_dict_emoji_sample = df_yaliwc_dict_emoji_sample[["items","weights",]]

def save__yaliwc_dict_emoji_sample( fn="yaliwc_dict_emoji_sample.csv", ):
    df_yaliwc_dict_emoji_sample.to_csv(fn,index=False,)


if __name__ == "__main__":
    save__yaliwc_dict_emoji_sample()
