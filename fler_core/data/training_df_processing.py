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

# Globals
###############################################################################

LOGGER = logzero.logger


# Functions and Classes
###############################################################################

class Training_database(object):
    """
    basic class to either, get the already trained database, or clean the dirty database
    """

    def __init__(self):
        self.df_name = None
        self.language = None
        self.categories = None
        self.df = None

    @classmethod
    async def get_already_trained(cls, name: str, language: str = cst.NO_LANGUAGE) -> Union[object, None]:
        """
        charge an already trained dataset, you must specify the langage
        :param langage: langage of the dataset
        :type langage: str
        :return: the Training database object
        :rtype: Union[object, None]
        """
        self = Training_database()
        self.df_name = name
        self.language = language
        cfg = get_asset_root()
        file = get_file_content(cfg, name)
        try:
            self.df = pd.read_csv(filepath_or_buffer=file, index_col=0)
            self.df = self.df.fillna(0)
            self.categories = list(self.df['NEtag'].unique())
            self.categories.remove(0)
            return self
        except:
            return "Error when trying to read the dataframe"

    @classmethod
    async def clean_and_setup_training(cls, name: str, language: str = cst.NO_LANGUAGE) -> Union[object, None]:
        """
        charge a virgin dataset, and add the features
        :param name: name of the virgin dataset
        :type name: str
        :param language: language of the dataset
        :type language: str
        :return: the new traning dataset object
        :rtype: Union[object, None]
        """
        self = Training_database
        self.df_name = name
        self.language = language
        cfg = get_asset_root()
        file = get_file_content(cfg, name)
        try:
            self.df = pd.read_csv(filepath_or_buffer=file, index_col=0)
        except:
            return "Error when trying to read the dataframe"



if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(Training_database.get_already_trained('dffrancais', 'fr'))
    LOGGER.info(ret.categories)

