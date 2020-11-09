#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from collections import OrderedDict

dict0_emoji_basic = OrderedDict()
dict0_emoji_basic[len(dict0_emoji_basic)] = [ ":*" ,  1 ]
dict0_emoji_basic[len(dict0_emoji_basic)] = [ "*:" ,  1 ]
dict0_emoji_basic[len(dict0_emoji_basic)] = [ ":)" ,  1 ]
dict0_emoji_basic[len(dict0_emoji_basic)] = [ "(:" ,  1 ]
dict0_emoji_basic[len(dict0_emoji_basic)] = [ ";)" ,  1 ]
dict0_emoji_basic[len(dict0_emoji_basic)] = [ "(;" ,  1 ]
dict0_emoji_basic[len(dict0_emoji_basic)] = [ ":(" , -1 ]
dict0_emoji_basic[len(dict0_emoji_basic)] = [ "):" , -1 ]
dict0_emoji_basic[len(dict0_emoji_basic)] = [ ":/" , -1 ]
dict0_emoji_basic[len(dict0_emoji_basic)] = [ "/:" , -1 ]


df0_emoji_basic = pd.DataFrame.from_dict(
    data    = dict0_emoji_basic,
    orient  = "index",
    columns = ["wildcard","weight",],
)
