#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from collections import OrderedDict

dict0_testing_words = OrderedDict()
dict0_testing_words[len(dict0_testing_words)] = [ "self"       ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "self-*"     ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "should*"    ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "skill"      ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "skilled"    ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "skillful*"  ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "skills*"    ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "wit"        ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "wits"       ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "won"        ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "won't"      ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "you"        ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "you'*"      ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "your"       ,   0.33 ]
dict0_testing_words[len(dict0_testing_words)] = [ "positiv*"   ,  10.00 ]
dict0_testing_words[len(dict0_testing_words)] = [ "negativ*"   , -10.00 ]

df0_testing_words = pd.DataFrame.from_dict(
    data    = dict0_testing_words,
    orient  = "index",
    columns = ["wildcard","weight",],
)


dict1_testing_docs = OrderedDict()
dict1_testing_docs[len(dict1_testing_docs)] = [ "doom with wit witted you're" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "(you're  self (self-taught) should't should not nee)" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "(::) :( :) :*" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "self self" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "self-taught self-less" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "self-taught self-" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "self selfless" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "self- you-yo" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "should shoulder SIC!" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "should shouldn't" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "shoulders should've" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "not-you'll" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "skill skillfulness" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "with wit witted" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "wit with witted" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "with witted wit" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "nothing-here" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "you :)" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "are you" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "you are" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "you are you" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "are you are" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "you'll test" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "test you'll" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "test you'll ok" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "you test YOU test you" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "you're" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "you'll you're you've" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "positive positivity" ]
dict1_testing_docs[len(dict1_testing_docs)] = [ "negative negativity" ]


df1_testing_docs = pd.DataFrame.from_dict(
    data    = dict1_testing_docs,
    orient  = 'index',
    columns = ['tweet_body',],
)
