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
# custom
from fler_utils.commons import get_asset_root, get_file_content
import fler_core.constants as cst
import fler_core.commons as com
# Globals
###############################################################################

LOGGER = logzero.logger


# Functions and Classes
###############################################################################

class Features(object):
    """
    class that adds all the features we want on the dataframe
    """

    def __init__(self):
        self.feature_list = None
        self.df = None

    @classmethod
    def apply_list_feature(cls, feature_list: list, df: pd.DataFrame, pre:str = None) -> Union[object, None]:
        """
        from a list of feature and a dataframe create a dataframe with the given features
        :param feature_list: list of the different features
        :type feature_list: list of strings
        :param df: the dataframe on which to apply the different features
        :type df: pd.DataFrame
        :return: the new dataframe updated
        :rtype: Union[object, None]
        """
        self = Features()
        self.df = df
        self.df = Features.lowercase(self.df)
        self.feature_list = [getattr(Features, ftr) for ftr in feature_list]
        for i in self.feature_list:
            self.df = i(self.df)
        return self

    @staticmethod
    def capitalize(df: pd.DataFrame):
        """
        feature that verifies if the first letter is capitalized
        :param df: the dataframe on which to apply the features
        :type df: pd.Dataframe
        :return: the updated Dataframe
        :rtype: pd.Dataframe
        """
        cap = []
        for x in df[cst.WORD]:
            if x[0].isupper():
                cap.append(1)
            else:
                cap.append(0)
        df[cst.CAP] = cap
        return df

    @staticmethod
    def fullcap(df: pd.DataFrame):
        """
        feature that checks if all the letter are capitalized or not
        :param df: the dataframe
        :type df: pd.Dataframe
        :return: updated dataframe
        :rtype: pd.Dataframe
        """
        full_cap = []
        for x in df[cst.WORD]:
            if str(x).isupper():
                full_cap.append(1)
            else:
                full_cap.append(0)
        df[cst.FULL_CAP] = full_cap
        return df

    @staticmethod
    def length(df: pd.DataFrame):
        """
        length of the word
        :param df: dataframe
        :type df: pd.Dataframe
        :return: updated dataframe
        :rtype: pd.Dataframe
        """
        length = []
        for x in df[cst.WORD]:
            length.append(len(x))
        df[cst.LENGTH] = length
        return df

    @staticmethod
    def lowercase(df: pd.DataFrame):
        """
        add a new columns with the words in lowercase
        :param df: dataframe
        :type df: pd.Dataframe
        :return: updated dataframe
        :rtype: pd.Dataframe
        """
        nocaps = []
        for i in df[cst.WORD]:
            nocaps.append(str(i).lower())
        df[cst.LOWERCASE] = nocaps
        return df

    @staticmethod
    def presufixe(df: pd.DataFrame, language: str='ENG'):
        """
        feature that fires if there is a suffixe or a prefix
        :param df: dataframe
        :type df: pd.DataFrame
        :param language: abr of the language
        :type language: str
        :return: updated df
        :rtype: pd.DataFrame
        """
        presuf = []
        prefixe, suffixe = com.get_presufix(language)
        for x in df[cst.LOWERCASE]:
            b = 0
            if len(x) > 5:
                for j in range(0, 6):
                    if x[0:j] in prefixe:
                        b = 1

                for j in range(0, 5):
                    if x[len(x) - j:len(x)] in suffixe:
                        b = 1
            presuf.append(b)
        df[cst.PRESUFIX]= presuf
        return df

    # @staticmethod
    # def gazetteer(df: pd.DataFrame, language: str='ENG'):



# if __name__ == '__main__':
