#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from collections import OrderedDict

dc_yaliwc_docs_dummy_sample = OrderedDict()

dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "doom with wit witted you're"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "(you're  self (self-taught) should't should not nee)"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "(::) :( :) :*"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "self self"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "self-taught self-less"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "self-taught self-"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "self selfless"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "self- you-yo"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "should shoulder SIC!"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "should shouldn't"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "shoulders should've"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "not-you'll"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "skill skillfulness"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "with wit witted"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "wit with witted"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "with witted wit"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "nothing-here"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "you :)"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "are you"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "you are"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "you are you"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "are you are"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "you'll test"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "test you'll"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "test you'll ok"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "you test YOU test you"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "you're"
dc_yaliwc_docs_dummy_sample[ len(dc_yaliwc_docs_dummy_sample.keys()) ] = "you'll you're you've"

df_yaliwc_docs_dummy_sample = pd.DataFrame.from_dict(
    data    = dc_yaliwc_docs_dummy_sample,
    orient  = 'index',
    columns = ['tweet_body',],
)

df_yaliwc_docs_dummy_sample["tweet_id"] = df_yaliwc_docs_dummy_sample.index
df_yaliwc_docs_dummy_sample = df_yaliwc_docs_dummy_sample[["tweet_id","tweet_body",]]

def save__yaliwc_docs_dummy_sample( fn="yaliwc_docs_dummy_sample.csv", ):
    df_yaliwc_docs_dummy_sample.to_csv(fn,index=False,)


if __name__ == "__main__":
    save__yaliwc_docs_dummy_sample()
