#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from collections import OrderedDict

dc_yaliwc_dict_big2b_commun = OrderedDict()

dc_yaliwc_dict_big2b_commun[ "affect*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "appreciat*"   ] = 1
dc_yaliwc_dict_big2b_commun[ "approach*"    ] = 1
dc_yaliwc_dict_big2b_commun[ "assist*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "benefit"      ] = 1
dc_yaliwc_dict_big2b_commun[ "care*"        ] = 1
dc_yaliwc_dict_big2b_commun[ "caring*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "chat*"        ] = 1
dc_yaliwc_dict_big2b_commun[ "close*"       ] = 1
dc_yaliwc_dict_big2b_commun[ "commun*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "compassion*"  ] = 1
dc_yaliwc_dict_big2b_commun[ "considerate*" ] = 1
dc_yaliwc_dict_big2b_commun[ "courteous*"   ] = 1
dc_yaliwc_dict_big2b_commun[ "decency"      ] = 1
dc_yaliwc_dict_big2b_commun[ "decent*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "dialogue*"    ] = 1
dc_yaliwc_dict_big2b_commun[ "emotion*"     ] = 1
dc_yaliwc_dict_big2b_commun[ "fair*"        ] = 1
dc_yaliwc_dict_big2b_commun[ "faith*"       ] = 1
dc_yaliwc_dict_big2b_commun[ "friend*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "generous*"    ] = 1
dc_yaliwc_dict_big2b_commun[ "genteel"      ] = 1
dc_yaliwc_dict_big2b_commun[ "gentle*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "gently"       ] = 1
dc_yaliwc_dict_big2b_commun[ "genuine*"     ] = 1
dc_yaliwc_dict_big2b_commun[ "grateful*"    ] = 1
dc_yaliwc_dict_big2b_commun[ "help*"        ] = 1
dc_yaliwc_dict_big2b_commun[ "honest*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "justice"      ] = 1
dc_yaliwc_dict_big2b_commun[ "justly"       ] = 1
dc_yaliwc_dict_big2b_commun[ "kindly"       ] = 1
dc_yaliwc_dict_big2b_commun[ "kindness"     ] = 1
dc_yaliwc_dict_big2b_commun[ "love*"        ] = 1
dc_yaliwc_dict_big2b_commun[ "loving*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "loyal*"       ] = 1
dc_yaliwc_dict_big2b_commun[ "mercy"        ] = 1
dc_yaliwc_dict_big2b_commun[ "modest*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "moral*"       ] = 1
dc_yaliwc_dict_big2b_commun[ "open*"        ] = 1
dc_yaliwc_dict_big2b_commun[ "patiently"    ] = 1
dc_yaliwc_dict_big2b_commun[ "polite*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "praise"       ] = 1
dc_yaliwc_dict_big2b_commun[ "protect*"     ] = 1
dc_yaliwc_dict_big2b_commun[ "righteous*"   ] = 1
dc_yaliwc_dict_big2b_commun[ "sense"        ] = 1
dc_yaliwc_dict_big2b_commun[ "share"        ] = 1
dc_yaliwc_dict_big2b_commun[ "shared"       ] = 1
dc_yaliwc_dict_big2b_commun[ "shares"       ] = 1
dc_yaliwc_dict_big2b_commun[ "sharing"      ] = 1
dc_yaliwc_dict_big2b_commun[ "socia*"       ] = 1
dc_yaliwc_dict_big2b_commun[ "solidarity"   ] = 1
dc_yaliwc_dict_big2b_commun[ "tact*"        ] = 1
dc_yaliwc_dict_big2b_commun[ "talk*"        ] = 1
dc_yaliwc_dict_big2b_commun[ "tender*"      ] = 1
dc_yaliwc_dict_big2b_commun[ "thank*"       ] = 1
dc_yaliwc_dict_big2b_commun[ "together*"    ] = 1
dc_yaliwc_dict_big2b_commun[ "tolerance"    ] = 1
dc_yaliwc_dict_big2b_commun[ "tolerant*"    ] = 1
dc_yaliwc_dict_big2b_commun[ "truth*"       ] = 1
dc_yaliwc_dict_big2b_commun[ "understand*"  ] = 1
dc_yaliwc_dict_big2b_commun[ "union*"       ] = 1
dc_yaliwc_dict_big2b_commun[ "unite*"       ] = 1
dc_yaliwc_dict_big2b_commun[ "unity"        ] = 1
dc_yaliwc_dict_big2b_commun[ "warm*"        ] = 1
dc_yaliwc_dict_big2b_commun[ "yield*"       ] = 1

df_yaliwc_dict_big2b_commun = pd.DataFrame.from_dict(
    data    = dc_yaliwc_dict_big2b_commun,
    orient  = 'index',
    columns = ['weights',],
)

df_yaliwc_dict_big2b_commun["items"] = df_yaliwc_dict_big2b_commun.index
df_yaliwc_dict_big2b_commun = df_yaliwc_dict_big2b_commun[["items","weights",]]

def save__yaliwc_dict_big2b_commun( fn="yaliwc_dict_big2b_commun.csv", ):
    df_yaliwc_dict_big2b_commun.to_csv(fn,index=False,)


if __name__ == "__main__":
    save__yaliwc_dict_big2b_commun()
