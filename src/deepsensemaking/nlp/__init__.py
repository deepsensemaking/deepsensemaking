#!/usr/bin/env python
# -*- coding: utf-8 -*

"""deepsensemaking (dsm) lang sub-module"""

import nltk

from collections import OrderedDict


POS_TAGSET = OrderedDict()
POS_TAGSET[ 'CC'   ] = 'Coordinating conjunction'
POS_TAGSET[ 'PRP$' ] = 'Possessive pronoun'
POS_TAGSET[ 'CD'   ] = 'Cardinal number'
POS_TAGSET[ 'RB'   ] = 'Adverb'
POS_TAGSET[ 'DT'   ] = 'Determiner'
POS_TAGSET[ 'RBR'  ] = 'Adverb, comparative'
POS_TAGSET[ 'EX'   ] = 'Existential there'
POS_TAGSET[ 'RBS'  ] = 'Adverb, superlative'
POS_TAGSET[ 'FW'   ] = 'Foreign word'
POS_TAGSET[ 'RP'   ] = 'Particle'
POS_TAGSET[ 'JJ'   ] = 'Adjective'
POS_TAGSET[ 'TO'   ] = 'to'
POS_TAGSET[ 'JJR'  ] = 'Adjective, comparative'
POS_TAGSET[ 'UH'   ] = 'Interjection'
POS_TAGSET[ 'JJS'  ] = 'Adjective, superlative'
POS_TAGSET[ 'VB'   ] = 'Verb, base form'
POS_TAGSET[ 'LS'   ] = 'List item marker'
POS_TAGSET[ 'VBD'  ] = 'Verb, past tense'
POS_TAGSET[ 'MD'   ] = 'Modal'
POS_TAGSET[ 'NNS'  ] = 'Noun, plural'
POS_TAGSET[ 'NN'   ] = 'Noun, singular or masps'
POS_TAGSET[ 'VBN'  ] = 'Verb, past participle'
POS_TAGSET[ 'VBZ'  ] = 'Verb,3rd ps. sing. present'
POS_TAGSET[ 'NNP'  ] = 'Proper noun, singular'
POS_TAGSET[ 'NNPS' ] = 'Proper noun plural'
POS_TAGSET[ 'WDT'  ] = 'wh-determiner'
POS_TAGSET[ 'PDT'  ] = 'Predeterminer'
POS_TAGSET[ 'WP'   ] = 'wh-pronoun'
POS_TAGSET[ 'POS'  ] = 'Possessive ending'
POS_TAGSET[ 'WP$'  ] = 'Possessive wh-pronoun'
POS_TAGSET[ 'PRP'  ] = 'Personal pronoun'
POS_TAGSET[ 'WRB'  ] = 'wh-adverb'
POS_TAGSET[ '('    ] = 'open parenthesis'
POS_TAGSET[ ')'    ] = 'close parenthesis'
POS_TAGSET[ '``'   ] = 'open quote'
POS_TAGSET[ ','    ] = 'comma'
POS_TAGSET[ "''"   ] = 'close quote'
POS_TAGSET[ '.'    ] = 'period'
POS_TAGSET[ '#'    ] = 'pound sign (currency marker)'
POS_TAGSET[ '$'    ] = 'dollar sign (currency marker)'
POS_TAGSET[ 'IN'   ] = 'Preposition/subord. conjunction'
POS_TAGSET[ 'SYM'  ] = 'Symbol (mathematical or scientific)'
POS_TAGSET[ 'VBG'  ] = 'Verb, gerund/present participle'
POS_TAGSET[ 'VBP'  ] = 'Verb, non-3rd ps. sing. present'
POS_TAGSET[ ':'    ] = 'colon'




def pos_cond_frq( pos_tuples_list, ):
    cond_freq_dist = nltk.ConditionalFreqDist((tag, word) for (word, tag) in pos_tuples_list)
    return cond_freq_dist
