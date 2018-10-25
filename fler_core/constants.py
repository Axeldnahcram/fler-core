# coding: utf-8

# Globals
###############################################################################

# commons

# commons on metadata dataframe
NO_LANGUAGE = 'No language specified'
NO_NAME = 'Default Name'

# commons on the features
CAP = 'Cap'
FULL_CAP = 'Full_Cap'
LENGTH = 'Length'
LOWERCASE = 'Lower_case'
PRESUFIX = 'Pre_suffixe'
# commons on the dataframe columns names
WORD = 'Word'

# commons on the prefix/suffix feature
ENG_PREFIX = ['anti', 'co', 'dis', 'il', 'im', 'in', 'inter', 'ir', 'mis', 'over', 'out', 'post', 'pre', 'pro', 'sub',
              'super', 'trans', 'under']
ENG_SUFFIX = ['dom', 'ship', 'hood', 'ian', 'er', 'er', 'ism', 'en', 'less', 'ish', 'ful', 'al', 'ly', 'en', 'ness',
              'ship', 'ity', 'ize', 'ly']
FR_PREFIX = ['a', 'an', 'ad', 'ac', 'dé', 'dis', 'é', 'in', 'im', 'irr', 'ill', 'mé', 'pré', 're', 'ré', 'co', 'col',
             'con', 'com', 'aéro', 'anti', 'auto', 'bi', 'di', 'ex', 'mal', 'para', 'pare', 'paro', 'post', 'néo',
             'pro', 'sub', 'suc', 'sug', 'sous', 'sou', 'sur', 'sus', 'tri', 'trans', 'hyper', 'hypo', 'syn', 'apo',
             'arch', 'archi', 'endo']
FR_SUFFIX = ['eur', 'euse', 'son', 'tion', 'ance', 'ment', 'ure', 'ade', 'age', 'aille', 'isme', 'iste', 'er', 'ère',
             'iste', 'eur', 'ien', 'ier', 'ie', 'ée', 'ain', 'ais', 'ois', 'ien', 'esse', 'ant', 'ain', 'ais', 'ois',
             'ien', 'able', 'ible', 'et', 'ot', 'ard', 'aud', 'iste', 'al', 'el', 'âtre', 'eur', 'eux', 'if', 'in']

# commons on categories
LOC = 'LOC'
PER = 'PER'
MISC = 'MISC'
ORG = 'ORG'

# list features
list_features_en = ['Cap', 'Full_Cap', 'Length', 'Pre_suffixe',
                 'GAZMISC', 'GAZPER', 'GAZLOC', 'Number', 'FreqNAMES', 'FreqORG',
                 'FreqLOC', 'FreqPER', 'FreqMISC', 'preuniORG', 'preuniLOC', 'preuniPER',
                 'preuniMISC', 'postuniORG', 'postuniLOC', 'postuniPER', 'postuniMISC', "debut"]

list_features_fr = ['Cap', 'Full_Cap', 'Length', 'Pre_suffixe', 'GAZPER', 'GAZLOC', 'Number', 'FreqNAMES', 'FreqORG',
                 'FreqLOC', 'FreqPER', 'preuniORG', 'preuniLOC', 'preuniPER', 'postuniORG', 'postuniLOC', 'postuniPER', "debut"]

list_features_fr_no_caps = ['Full_Cap', 'Length', 'Pre_suffixe', 'GAZPER', 'GAZLOC', 'Number', 'FreqNAMES', 'FreqORG',
                 'FreqLOC', 'FreqPER', 'preuniORG', 'preuniLOC', 'preuniPER', 'postuniORG', 'postuniLOC', 'postuniPER', "debut"]
