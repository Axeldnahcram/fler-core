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
import os
import numpy as np
import logzero
# custom
from fler_utils.commons import get_asset_root, get_file_content, get_type_of_gazetteers
import fler_core.constants as cst
import fler_core.commons as com
import collections
import jsonpickle
import json

# Globals
###############################################################################

LOGGER = logzero.logger


# Functions and Classes
###############################################################################

class Feature_fr(object):
    """
    class that adds all the Feature_eng we want on the dataframe
    """

    def __init__(self):
        self.feature_list = None
        self.df = None

    @classmethod
    def apply_list_feature(cls, feature_list: list, df: pd.DataFrame, pre: str = None) -> Union[object, None]:
        """
        from a list of feature and a dataframe create a dataframe with the given Feature_eng
        :param feature_list: list of the different Feature_eng
        :type feature_list: list of strings
        :param df: the dataframe on which to apply the different Feature_eng
        :type df: pd.DataFrame
        :return: the new dataframe updated
        :rtype: Union[object, None]
        """
        self = Feature_fr()
        self.df = df
        self.df = Feature_fr.lowercase(self.df)
        self.feature_list = [getattr(Feature_fr, ftr) for ftr in feature_list]
        for i in self.feature_list:
            self.df = i(self.df)
        return self

    @staticmethod
    def capitalize(df: pd.DataFrame):
        """
        feature that verifies if the first letter is capitalized
        :param df: the dataframe on which to apply the Feature_eng
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
    def presufixe(df: pd.DataFrame, language: str = 'FR'):
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
        df[cst.PRESUFIX] = presuf
        return df

    @staticmethod
    def gazetteer(df: pd.DataFrame, language: str = 'FR'):
        cfg = get_asset_root()

        list_gaz = get_type_of_gazetteers(cfg, 'fr')
        for i in list_gaz:
            g = [0 for i in df[cst.WORD]]
            list_files = get_file_content(cfg, 'gazetteer_fr', gaztype=i)
            for j in list_files:
                gaz = list(pd.read_csv(j)[cst.LOWERCASE])

                for index in range(0, len(df[cst.LOWERCASE])):
                    if df[cst.LOWERCASE][index] in gaz:
                        g[index] = 1
            df[i] = g
        return df

    @staticmethod
    def frequency_train(df: pd.DataFrame, liste_NP: list = ["ORG", "LOC", "PER"]):
        allname = []
        json_final_file = {}
        for row in df.itertuples(index=True, name='Pandas'):
            if getattr(row, 'NEtag') != "O":
                allname.append(getattr(row, cst.LOWERCASE))
        counter = collections.Counter(allname)
        frequentname = []
        for i in allname:
            if counter[i] >= 6 and i not in frequentname:
                frequentname.append(i)
        frequentname = {"FreqNAMES":frequentname}
        freq = []
        with open(f'freqNAMES.json', 'w') as outfile:
            json.dump(frequentname, outfile)
        for row in df.itertuples(index=True, name='Pandas'):
            if getattr(row, cst.LOWERCASE) in frequentname:
                freq.append(1)
            else:
                freq.append(0)
        df['FreqNAMES'] = freq
        for type_NP in liste_NP:
            allname = []
            for row in df.itertuples(index=True, name='Pandas'):
                if getattr(row, type_NP) != 0:
                    allname.append(getattr(row, cst.LOWERCASE))
            counter = collections.Counter(allname)

            frequentname = []
            for i in allname:
                if counter[i] >= 6 and i not in frequentname:
                    frequentname.append(i)
            frequentname = {f"Freq{type_NP}":frequentname}
            with open(f'freq{type_NP}.json', 'w') as outfile:
                json.dump(frequentname, outfile)

            freq = []
            for row in df.itertuples(index=True, name='Pandas'):
                if getattr(row, cst.LOWERCASE) in frequentname:
                    freq.append(1)
                else:
                    freq.append(0)
            df[f'Freq{type_NP}'] = freq
        return df

    @staticmethod
    def frequency_factory(df: pd.DataFrame, directory:str="freq_names_french"):
        freq = {'FreqNAMES': f'{directory}/freqNAMES', 'FreqORG': f'{directory}/freqORG',
                'FreqLOC':   f'{directory}/freqLOC', 'FreqPER': f'{directory}/freqPER'}
        cfg = get_asset_root()
        for key, value in freq.items():
            file_name = get_file_content(cfg, value)
            with open(file_name) as json_file:
                data = json.load(json_file)
            frequentname = data[key]
            LOGGER.info(frequentname)
            freq_entity = []
            for row in df.itertuples(index=True, name='Pandas'):
                if getattr(row, cst.LOWERCASE) in frequentname:
                    freq_entity.append(1)
                else:
                    freq_entity.append(0)
            df[key] = freq_entity
        return df

    @staticmethod
    def number(df: pd.DataFrame):
        Numb = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        nube = []
        for row in df.itertuples(index=True, name='Pandas'):
            word = getattr(row, cst.LOWERCASE)
            if word == 0:
                nube.append(0)
            else:
                g = 0
                for i in Numb:
                    if i in word:
                        g = 1
                nube.append(g)
        df["Number"] = nube
        return df

    @staticmethod
    def preuni_train(df: pd.DataFrame, liste_NP: list = ["ORG", "LOC", "PER"]):
        for at in liste_NP:
            LOGGER.info(at)
            uni = []
            for i, row in enumerate(df.itertuples(index=True, name='Pandas'), 1):
                if getattr(row, at) == 1:
                    uni.append(df.iloc[i - 2, :][cst.LOWERCASE])
            counter2 = collections.Counter(uni)

            frequentuni = []
            for i in uni:
                if counter2[i] >= 6 and i not in frequentuni:
                    frequentuni.append(i)
            with open(f'preuni{at}.json', 'w') as outfile:
                json.dump({f'preuni{at}': frequentuni}, outfile)
            L = [0]
            for i in range(1, len(df)):
                if df.iloc[i - 1][cst.LOWERCASE] in frequentuni:
                    L.append(1)
                else:
                    L.append(0)
            df['PREUNI' + at] = L
        return df

    @staticmethod
    def postuni_train(df: pd.DataFrame, liste_NP: list = ["ORG", "LOC", "PER"]):
        for at in liste_NP:
            uni = []
            for i, row in enumerate(df.itertuples(index=True, name='Pandas'), 1):
                if getattr(row, at) == 1:
                    uni.append(df.iloc[i - 2, :][cst.LOWERCASE])
            counter2 = collections.Counter(uni)

            frequentuni = []
            for i in uni:
                if counter2[i] >= 6 and i not in frequentuni:
                    frequentuni.append(i)
            with open(f'postuni{at}.json', 'w') as outfile:
                json.dump({f'postuni{at}': frequentuni}, outfile)
            L = []
            for i in range(0, len(df) - 1):
                if df.iloc[i + 1][cst.LOWERCASE] in frequentuni:
                    L.append(1)
                else:
                    L.append(0)
            L.append(0)
            df['POSTUNI' + at] = L
        return df

    @staticmethod
    def preuni_factory(df: pd.DataFrame, directory:str="pre_freq_french"):
        preuni = {'preuniORG':  f'{directory}/preuniORG',
                  'preuniLOC':  f'{directory}/preuniLOC', 'preuniPER': f'{directory}/preuniPER'}
        cfg = get_asset_root()
        for key, value in preuni.items():
            file_name = get_file_content(cfg, value)
            with open(file_name) as json_file:
                data = json.load(json_file)
            frequentname = data[key]
            LOGGER.info(frequentname)
            L = [0]
            for i in range(1, len(df)):
                if df.iloc[i - 1][cst.LOWERCASE] in frequentname:
                    L.append(1)
                else:
                    L.append(0)
            df[key] = L
        return df

    @staticmethod
    def postuni_factory(df: pd.DataFrame, directory:str="post_freq_french"):
        postuni = {'postuniORG':  f'{directory}/postuniORG',
                   'postuniLOC':  f'{directory}/postuniLOC', 'postuniPER': f'{directory}/postuniPER'}
        cfg = get_asset_root()
        for key, value in postuni.items():
            LOGGER.info(value)
            file_name = get_file_content(cfg, value)
            LOGGER.info(file_name)
            with open(file_name) as json_file:
                data = json.load(json_file)
            frequentname = data[key]
            LOGGER.info(frequentname)
            L = []
            for i in range(0, len(df) - 1):
                if df.iloc[i + 1][cst.LOWERCASE] in frequentname:
                    L.append(1)
                else:
                    L.append(0)
            L.append(0)
            df[key] = L
        return df

    @staticmethod
    def debut(df: pd.DataFrame):
        freq = [0]
        for i in range(1, len(df)):
            if df.iloc[i - 1][cst.LOWERCASE] in ['.','!', '?']:
                freq.append(1)
            else:
                freq.append(0)
        df['debut'] = freq
        return df

features_no_directory_fr = [Feature_fr.lowercase, Feature_fr.capitalize, Feature_fr.fullcap,
                         Feature_fr.length, Feature_fr.presufixe, Feature_fr.gazetteer, Feature_fr.number, Feature_fr.debut]
features_no_directory_no_casp_fr=[Feature_fr.lowercase, Feature_fr.fullcap,
                         Feature_fr.length, Feature_fr.presufixe, Feature_fr.gazetteer, Feature_fr.number, Feature_fr.debut]

if __name__ == "__main__":
    cfg = get_asset_root()
    directory = get_file_content(cfg, "French_own_data/Reuters_with_NE_columns")
    LOGGER.info(directory)
    df = pd.read_csv(directory)
    df = Feature_fr.lowercase(df)
    df = Feature_fr.gazetteer(df)

    LOGGER.info(df)