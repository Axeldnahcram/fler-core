# coding: utf-8
"""
.. module:: fler_core.training_df.py
basic class for a training database
"""

__author__ = "Axel Marchand"

# standard
import asyncio
from typing import Union, List, Dict
import pandas as pd
import numpy as np
import logzero
#installed
import nltk
#custom
from fler_utils.commons import get_asset_root, get_file_content
import fler_core.constants as cst

# Globals
###############################################################################

LOGGER = logzero.logger

# Functions and Classes
###############################################################################

def get_presufix (language : str='ENG'):
    """
    get the prefix and the suffix list for a given language (default english)
    :param language: abr of the language
    :type language: str
    :return: two lists
    :rtype: list
    """
    if language == 'ENG':
        return cst.ENG_PREFIX, cst.ENG_SUFFIX
    if language == 'FR':
        return cst.FR_PREFIX, cst.FR_SUFFIX

def no_caps(text):
    nocaps=[]
    for i in text :
        nocaps.append(str(i).lower())
    return nocaps

def get_gazetteer(language: str='ENG'):
    """
    return a dictionary with the types and the gazetteers associated
    :param language: language in wich we try to get the gazetteers
    :type language: str
    :return: dictionary {'LOC': [Paris, ...]}
    :rtype: dict
    """
    if language == 'ENG':
        gazloc = no_caps(nltk.corpus.gazetteers.words(
            fileids=['countries.txt', 'uscities.txt', 'usstates.txt', 'usstateabbrev.txt', 'mexstates.txt',
                     'caprovinces.txt']))
        gazper = no_caps(nltk.corpus.names.words(fileids=['male.txt', 'female.txt']))
        gazmisc = no_caps(nltk.corpus.gazetteers.words(fileids=['nationalities.txt']))
        return {cst.LOC: gazloc, cst.PER: gazper, cst.MISC: gazmisc}
    if language == 'FR':
        cfg = get_asset_root()
        gazloc = get_file_content(cfg, 'gazLOC')
        gazloc = pd.read_csv(gazloc)
        gazloc = gazloc.iloc[:,0].tolist()
        gazper = get_file_content(cfg, 'gazPER')
        gazper = pd.read_csv(gazper)
        gazper = gazper.iloc[:, 0].tolist()
        return {cst.LOC: gazloc, cst.PER: gazper}

if __name__ == '__main__':
    f = get_gazetteer('ENG')
    LOGGER.info(f)

