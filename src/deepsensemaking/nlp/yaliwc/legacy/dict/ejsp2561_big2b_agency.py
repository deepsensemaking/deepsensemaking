#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from collections import OrderedDict

dc_yaliwc_dict_big2b_agency = OrderedDict()

dc_yaliwc_dict_big2b_agency[ "able"            ] = 1
dc_yaliwc_dict_big2b_agency[ "accomplish*"     ] = 1
dc_yaliwc_dict_big2b_agency[ "achiev*"         ] = 1
dc_yaliwc_dict_big2b_agency[ "activ*"          ] = 1
dc_yaliwc_dict_big2b_agency[ "advance*"        ] = 1
dc_yaliwc_dict_big2b_agency[ "aggressive"      ] = 1
dc_yaliwc_dict_big2b_agency[ "ambition*"       ] = 1
dc_yaliwc_dict_big2b_agency[ "ambitious*"      ] = 1
dc_yaliwc_dict_big2b_agency[ "assert*"         ] = 1
dc_yaliwc_dict_big2b_agency[ "attain*"         ] = 1
dc_yaliwc_dict_big2b_agency[ "authoritative.*" ] = 1
dc_yaliwc_dict_big2b_agency[ "autonomous*"     ] = 1
dc_yaliwc_dict_big2b_agency[ "autonomy"        ] = 1
dc_yaliwc_dict_big2b_agency[ "brave*"          ] = 1
dc_yaliwc_dict_big2b_agency[ "capab*"          ] = 1
dc_yaliwc_dict_big2b_agency[ "command*"        ] = 1
dc_yaliwc_dict_big2b_agency[ "confidence"      ] = 1
dc_yaliwc_dict_big2b_agency[ "confident"       ] = 1
dc_yaliwc_dict_big2b_agency[ "control"         ] = 1
dc_yaliwc_dict_big2b_agency[ "controlling"     ] = 1
dc_yaliwc_dict_big2b_agency[ "courage*"        ] = 1
dc_yaliwc_dict_big2b_agency[ "creat*"          ] = 1
dc_yaliwc_dict_big2b_agency[ "dare*"           ] = 1
dc_yaliwc_dict_big2b_agency[ "decision*"       ] = 1
dc_yaliwc_dict_big2b_agency[ "decisive*"       ] = 1
dc_yaliwc_dict_big2b_agency[ "determin*"       ] = 1
dc_yaliwc_dict_big2b_agency[ "discover*"       ] = 1
dc_yaliwc_dict_big2b_agency[ "dominant*"       ] = 1
dc_yaliwc_dict_big2b_agency[ "dynamic"         ] = 1
dc_yaliwc_dict_big2b_agency[ "eager*"          ] = 1
dc_yaliwc_dict_big2b_agency[ "effective*"      ] = 1
dc_yaliwc_dict_big2b_agency[ "excellent*"      ] = 1
dc_yaliwc_dict_big2b_agency[ "experience*"     ] = 1
dc_yaliwc_dict_big2b_agency[ "expert"          ] = 1
dc_yaliwc_dict_big2b_agency[ "independ*"       ] = 1
dc_yaliwc_dict_big2b_agency[ "influence"       ] = 1
dc_yaliwc_dict_big2b_agency[ "inform*"         ] = 1
dc_yaliwc_dict_big2b_agency[ "intelligence"    ] = 1
dc_yaliwc_dict_big2b_agency[ "intelligent*"    ] = 1
dc_yaliwc_dict_big2b_agency[ "intent*"         ] = 1
dc_yaliwc_dict_big2b_agency[ "know*"           ] = 1
dc_yaliwc_dict_big2b_agency[ "lead*"           ] = 1
dc_yaliwc_dict_big2b_agency[ "logic*"          ] = 1
dc_yaliwc_dict_big2b_agency[ "manager*"        ] = 1
dc_yaliwc_dict_big2b_agency[ "organized"       ] = 1
dc_yaliwc_dict_big2b_agency[ "outstanding*"    ] = 1
dc_yaliwc_dict_big2b_agency[ "overcome"        ] = 1
dc_yaliwc_dict_big2b_agency[ "power*"          ] = 1
dc_yaliwc_dict_big2b_agency[ "practic*"        ] = 1
dc_yaliwc_dict_big2b_agency[ "pride"           ] = 1
dc_yaliwc_dict_big2b_agency[ "productive*"     ] = 1
dc_yaliwc_dict_big2b_agency[ "professional*"   ] = 1
dc_yaliwc_dict_big2b_agency[ "proud*"          ] = 1
dc_yaliwc_dict_big2b_agency[ "rational*"       ] = 1
dc_yaliwc_dict_big2b_agency[ "reasoning"       ] = 1
dc_yaliwc_dict_big2b_agency[ "scientific*"     ] = 1
dc_yaliwc_dict_big2b_agency[ "skill"           ] = 1
dc_yaliwc_dict_big2b_agency[ "status"          ] = 1
dc_yaliwc_dict_big2b_agency[ "strength"        ] = 1
dc_yaliwc_dict_big2b_agency[ "strong*"         ] = 1
dc_yaliwc_dict_big2b_agency[ "succeed*"        ] = 1
dc_yaliwc_dict_big2b_agency[ "success*"        ] = 1
dc_yaliwc_dict_big2b_agency[ "thought*"        ] = 1
dc_yaliwc_dict_big2b_agency[ "triumph*"        ] = 1


df_yaliwc_dict_big2b_agency = pd.DataFrame.from_dict(
    data    = dc_yaliwc_dict_big2b_agency,
    orient  = 'index',
    columns = ['weights',],
)

df_yaliwc_dict_big2b_agency["items"] = df_yaliwc_dict_big2b_agency.index
df_yaliwc_dict_big2b_agency = df_yaliwc_dict_big2b_agency[["items","weights",]]

def save__yaliwc_dict_big2b_agency( fn="yaliwc_dict_big2b_agency.csv", ):
    df_yaliwc_dict_big2b_agency.to_csv(fn,index=False,)


if __name__ == "__main__":
    save__yaliwc_dict_big2b_agency()
