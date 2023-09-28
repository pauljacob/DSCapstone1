import pandas as pd
import os
import sqlite3
import numpy as np
import pickle
from functools import reduce


#plots
import matplotlib.pyplot as plt
import seaborn as sns

class color:
   BOLD = '\033[1m'
   END = '\033[0m'

#modeling
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn import __version__ as sklearn_version
import datetime
import pickle


# !pip install geopy
# !pip install Nominatim
# !pip install haversine
# !pip install timezonefinder

#feature engineering
from geopy.geocoders import Nominatim
import haversine as hs
from haversine import Unit

from timezonefinder import TimezoneFinder
from pytz import timezone

from sklearn.model_selection import TimeSeriesSplit


def initialize_custom_notebook_settings():
    '''
    initialize the jupyter notebook display width'''
        
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))

    
    pd.options.display.max_columns = 999
    pd.options.display.max_rows = 999

    pd.set_option('display.max_colwidth', None)
    pd.options.display.max_info_columns = 999


'''
Convenience functions: read, sort, print, and save data frame or dictionary.
'''
def p(df):
    '''
    Return the first 5 and last 5 rows of this DataFrame.'''
    if df.shape[0] > 6:
        print(df.shape)
        return pd.concat([df.head(), df.tail()])
    else:
        return df

def rcp(filename, parse_dates=None, index_col=None):
    '''
    Read a file from the processed_data folder.'''
    if parse_dates == None:
        return pd.read_csv(os.path.join('..', '03_processed_data', filename), index_col=index_col)
    else:
        return pd.read_csv(os.path.join('..', '03_processed_data', filename), parse_dates=parse_dates,  index_col=index_col)

def rpp(filename, parse_dates=None):
    '''
    Save collection and return it.'''

    relative_file_path = os.path.join('..', '03_processed_data', filename)
        
    with (open(relative_file_path, "rb")) as open_file:
        return pickle.load(open_file)
    
def rcr(filename, parse_dates=None):
    '''
    Read a file from the raw_data folder.'''
    if parse_dates == None:
        return pd.read_csv(os.path.join('..', '02_raw_data', filename))
    else:
        return pd.read_csv(os.path.join('..', '02_raw_data', filename), parse_dates=parse_dates)

def sr(df, column_name_list):
    '''
    Sort DataFrame by column(s) and reset its index.'''
    df = df.sort_values(column_name_list)
    return df.reset_index(drop=True)

def pl(list_):
    '''
    Print the list length and return the list.'''
    print(len(list_))
    return list_

def pdc(dict_):
    '''
    Print the dictionary length and return the dictionary.'''
    print(len(dict_))
    return dict_

def save_and_return_data_frame(df, filename, index=False, parse_dates=False, index_label=None):
    '''
    Save data frame and return it.'''
    
    relative_directory_path = os.path.join('..', '03_processed_data')

    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)

    relative_file_path = os.path.join(relative_directory_path, filename)
    if not os.path.exists(relative_file_path):
        df.to_csv(relative_file_path, index=index, index_label=index_label)
    elif os.path.exists(relative_file_path):
        print('This file already exists.')
        
    return rcp(filename, parse_dates, index_col=index_label)


def save_and_return_collection(data_frame_collection,
                               filename,
                               index=False,
                               parse_dates=None):
    '''
    Save collection and return it.'''

    relative_directory_path = os.path.join('..', '03_processed_data')

    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)

    relative_file_path = os.path.join(relative_directory_path, filename)
    
    if not os.path.exists(relative_file_path):
        file_object_wb =  open(relative_file_path, "wb")
        pickle.dump(data_frame_collection, file_object_wb)
        file_object_wb.close()

    elif os.path.exists(relative_file_path):
        print('This file already exists.')
        
    with (open(relative_file_path, "rb")) as open_file:
        data_frame_collection_readback = pickle.load(open_file)

    return data_frame_collection_readback


def return_processed_data_file_if_it_exists(filename,
                                            parse_dates=False):
    
    relative_directory_path = os.path.join('..', '03_processed_data')

    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)

    relative_file_path = os.path.join(relative_directory_path, filename)

    if os.path.exists(relative_file_path):
        print('This file already exists')
        return rcp(filename, parse_dates)
    else:
        return pd.DataFrame({})

    
def return_processed_collection_if_it_exists(filename,
                                             parse_dates=False):
    import pickle
    
    relative_directory_path = os.path.join('..', '03_processed_data')

    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)

    relative_file_path = os.path.join(relative_directory_path, filename)

    if os.path.exists(relative_file_path):
        print('This file already exists')
        
        with (open(relative_file_path, "rb")) as openfile:
            data_frame_collection_readback = pickle.load(openfile)
        
        return data_frame_collection_readback
    
    else:
        return None
    
    
def return_figure_if_it_exists(filename):

    import glob
    import imageio

    image = None
    for image_path in glob.glob(filename):
        image = imageio.imread(image_path)
        
    return image

    

def show_data_frames_in_memory(dir_):
    alldfs = [var for var in dir_ if isinstance(eval(var), pd.core.frame.DataFrame)]

    print(alldfs)


    
def get_column_name_list_left_not_in_right(df_left,
                                           df_right):

    column_name_list_in_both = list(set(df_left.columns).intersection(set(df_right.columns)))

    return [k for k in df_left.columns if k not in column_name_list_in_both]



    
    
    
    
    
##########################################################################################################################
'''
Data Wrangling functions
'''
##########################################################################################################################
def merge_data_frame_list(data_frame_list):
    return reduce(lambda left, right : pd.merge(left,
                                                right,
                                                on=[column_name for column_name in left if column_name in right],
                                                how='inner'), 
                  data_frame_list)



def get_game_date_game_id_1():
    df = pd.read_csv(filepath_or_buffer=os.path.join('..', '02_raw_data', 'games.csv'), 
                     parse_dates=['GAME_DATE_EST'])

    df = df.sort_values(['GAME_DATE_EST', 'GAME_ID'])

    df = df.loc[(df.loc[:, 'GAME_DATE_EST'].dt.year > 2005) & (df.loc[:, 'GAME_DATE_EST'].dt.year < 2018), :]

    df = df.reset_index(drop=True)

    df.rename(columns={'GAME_DATE_EST':'game_date', 'GAME_ID':'game_id'}, inplace=True)

    df_game_date_game_id = df.loc[:, ['game_date', 'game_id']]
    
    return df_game_date_game_id


def read_clean_and_filter_nba_box_scores_2(filename,
                                           season_list,
                                           season_type_list):
    
    #read file
    df = rcr(filename, ['game_date'])
    df = df.drop(columns=['matchup', 'w', 'l', 'a_team_id'])


    #filter data frame
    df = df.loc[df.loc[:, 'season'].isin(season_list) & df.loc[:, 'season_type'].isin(season_type_list)]
    
    #clean data frame
    df.loc[:, 'ft_pct'] = df.loc[:, 'ftm'] / df.loc[:, 'fta']
    
    
    column_name_list_int64 = [k for k in list(df.select_dtypes('number').columns) if not k in ['w_pct', 'min', 'fg_pct', 'fg3_pct', 'ft_pct']]
    
    df.loc[:, column_name_list_int64] = \
    df.loc[:, column_name_list_int64].astype('int64')
    
    df = df.loc[~(df.loc[:, 'min'] == 0), :]
    
    df = df.sort_values(['game_date', 'game_id'])
    df = df.reset_index(drop=True)
    
    return df



def fill_team_box_scores_game_date_2_1(df0, df1):

    df = pd.merge(df0, df1, how='left', on='game_id')

    df.loc[:, 'game_date_x'] = df.loc[:, 'game_date_x'].fillna(df.loc[:, 'game_date_y'])

    df = df.drop(columns='game_date_y').rename(columns={'game_date_x':'game_date'})
    
    df = df.sort_values(['game_date', 'game_id'])
    
    df = df.reset_index(drop=True)
    
    return df




def rename_columns_by_data_type(df,
                                exclude_column_name_list,
                                indicator_column_name,
                                indicator_column_value,
                                number_column_added_suffix,
                                data_type):    
    '''
    select and rename number data type columns with added suffix.'''


    df_team_box_scores_2_1_data_type_column_suffix_name = \
    df.loc[df.loc[:, indicator_column_name] == indicator_column_value, :].drop(columns=[indicator_column_name])

    column_name_list = \
    list(df_team_box_scores_2_1_data_type_column_suffix_name.select_dtypes(data_type).drop(columns=exclude_column_name_list).columns)

    column_name_list_suffix_added = \
    [k + number_column_added_suffix for k in column_name_list]

    column_name_dict_suffix_added = \
    dict(zip(column_name_list, column_name_list_suffix_added))

    return df_team_box_scores_2_1_data_type_column_suffix_name.rename(columns=column_name_dict_suffix_added), column_name_list_suffix_added



def transform_box_scores_2_1_to_away_home_columns(df_team_box_scores_2_1):
    
    def transform_box_scores_2_1_number_columns_to_away_home_columns(df_team_box_scores_2_1=df_team_box_scores_2_1):
        
        #get number away and number home data frames
        df_team_box_scores_2_1_number_away, column_name_list_number_away = \
        rename_columns_by_data_type(df=df_team_box_scores_2_1,
                                    exclude_column_name_list=['game_id', 'season_year'],
                                    indicator_column_name='is_home',
                                    indicator_column_value='f',
                                    number_column_added_suffix='_away',
                                    data_type='number')

        df_team_box_scores_2_1_number_home, column_name_list_number_home = \
        rename_columns_by_data_type(df=df_team_box_scores_2_1,
                                    exclude_column_name_list=['game_id', 'season_year'],
                                    indicator_column_name='is_home',
                                    indicator_column_value='t',
                                    number_column_added_suffix='_home',
                                    data_type='number')

        #add df_team_box_scores_2_1_number_away away box score column names to df_team_box_scores_2_1_home with values 0.
        df_team_box_scores_2_1_number_home.loc[:, column_name_list_number_away] = 0

        #add df_team_box_scores_2_1_number_home home box score column names to df_team_box_scores_2_1_away with value 0.
        df_team_box_scores_2_1_number_away.loc[:, column_name_list_number_home] = 0

        #combine number home and number away data frames
        df_team_box_scores_2_1_number_away_home = \
        pd.concat([df_team_box_scores_2_1_number_away, 
                   df_team_box_scores_2_1_number_home]).groupby(['game_id', 'game_date', 'season_year', 
                                                             'season_type', 'season']).sum().reset_index()

        return sr(df_team_box_scores_2_1_number_away_home, ['game_date', 'game_id'])
        
        
    def transform_box_scores_2_1_object_columns_to_away_home_columns(df_team_box_scores_2_1=df_team_box_scores_2_1):
        
        #get object home and object away data frames
        df_team_box_scores_2_1_object_away, column_name_list_object_away = \
        rename_columns_by_data_type(df=df_team_box_scores_2_1,
                                    exclude_column_name_list=['season_type', 'season'],
                                    indicator_column_name='is_home',
                                    indicator_column_value='f',
                                    number_column_added_suffix='_away',
                                    data_type='object')

        df_team_box_scores_2_1_object_home, column_name_list_object_home = \
        rename_columns_by_data_type(df=df_team_box_scores_2_1,
                                    exclude_column_name_list=['season_type', 'season'],
                                    indicator_column_name='is_home',
                                    indicator_column_value='t',
                                    number_column_added_suffix='_home',
                                    data_type='object')

        #combine object columns
        df_team_box_scores_2_1_object_away_home = pd.merge(df_team_box_scores_2_1_object_home.loc[:, ['game_id', 'game_date', 'season_year', 'season_type', 'season', 'wl_home']], df_team_box_scores_2_1_object_away.loc[:, ['game_id', 'game_date', 'season_year', 'season_type', 'season', 'wl_away']], on=['game_id', 'game_date', 'season_year', 'season_type', 'season'], how='inner')

        return sr(df_team_box_scores_2_1_object_away_home, ['game_date', 'game_id'])
    
    
    
    
    
    #get number columns as away and home columns
    df_team_box_scores_2_1_number_away_home = transform_box_scores_2_1_number_columns_to_away_home_columns(df_team_box_scores_2_1=df_team_box_scores_2_1)    

    
    #get object columns as away and home columns
    df_team_box_scores_2_1_object_away_home = transform_box_scores_2_1_object_columns_to_away_home_columns(df_team_box_scores_2_1=df_team_box_scores_2_1)
    
    
    
    #combine number columns with object columns
    df_team_box_scores_2_1_away_home = \
    pd.concat([df_team_box_scores_2_1_number_away_home, 
               df_team_box_scores_2_1_object_away_home.loc[:, ['wl_away', 'wl_home']]],
              axis=1)
    
    
    return sr(df_team_box_scores_2_1_away_home, ['game_date', 'game_id'])



def extract_and_add_score_difference(df):
    
    df.loc[:, 'score_difference_away'] = \
    df.loc[:, 'pts_home'].values - df.loc[:, 'pts_away']
    
    df.loc[:, 'score_difference_home'] = \
    df.loc[:, 'pts_away'].values - df.loc[:, 'pts_home']
    
    return df
   



def filter_data_frame_by_season(df, season_list):
    
    df = df.loc[df.loc[:,'season'].isin(season_list), :]

    return sr(df, ['game_date', 'game_id'])





#get time series cross validation 9 splits test game_id
def get_time_series_cross_validation_9_splits_test_game_id(df):

    column_name_list = df.columns.to_list()
    if 'game_id' in column_name_list and 'game_date' in column_name_list:
        df = df.sort_values(['game_date', 'game_id']).reset_index(drop=True)
    del column_name_list

    from sklearn.model_selection import TimeSeriesSplit
    time_series_cross_validation_9_splits = TimeSeriesSplit(max_train_size=None, n_splits=9)

    df_test_collection = {}

    for i, (train_indices, test_indices) in enumerate(time_series_cross_validation_9_splits.split(df)):    
        df_test_collection[i] = df.iloc[test_indices, :]

    #convert data frame collection to data frame
    df_time_series_cross_validation_9_splits_test_game_id_game_date = pd.DataFrame()
    for i in range(len(df_test_collection)):
        df_time_series_cross_validation_9_splits_test_game_id_game_date = \
        pd.concat([df_time_series_cross_validation_9_splits_test_game_id_game_date, df_test_collection[i]])

    #reset index
    df_time_series_cross_validation_9_splits_test_game_id_game_date = \
    df_time_series_cross_validation_9_splits_test_game_id_game_date.reset_index(drop=True)

    return df_time_series_cross_validation_9_splits_test_game_id_game_date




def add_spread_to_box_scores_away_home_for_select_seasons(df_team_box_scores_2_1_away_home, df_sportsbook_spread, season_list):
    '''
    comment.'''
        
    #get at least one row for each game_id
    df_sportsbook_spread = df_sportsbook_spread.iloc[df_sportsbook_spread.game_id.drop_duplicates().index].reset_index(drop=True)

    #filter for NBA box scores away home for the 2010-11 thru 2017-18 seasons
    df_team_box_scores_2_1_away_home_2010_11_2017_18 = df_team_box_scores_2_1_away_home.loc[df_team_box_scores_2_1_away_home.loc[:, 'season'].isin(season_list), :]
    
    #add spread dataset to NBA box scores away home for the 2010-11 thru 2017-18 seasons.
    column_name_dict_team_id_away_home = {'a_team_id':'team_id_home', 'team_id':'team_id_away'}

    df_team_box_scores_2_1_away_home_2010_11_2017_18_spread_3 = pd.merge(df_team_box_scores_2_1_away_home_2010_11_2017_18, df_sportsbook_spread.rename(columns=column_name_dict_team_id_away_home), on=['game_id', 'team_id_home', 'team_id_away'], how='inner')

    return df_team_box_scores_2_1_away_home_2010_11_2017_18_spread_3





def get_and_combine_player_advanced_box_scores():

    #get the splits of player advanced box scores
    filename_list = []
    for file in os.listdir(os.path.join('..', '04_processed_data_preliminary')):
        if file.startswith('5game_id_team_id_player_name_advanced_stats'):
            filename_list = filename_list + [str(file)]

    #combine the player advanced box scores
    df_list = []

    for i in range(len(filename_list)):
        df_list = \
        df_list + [pd.read_csv(os.path.join('..', '04_processed_data_preliminary', str(filename_list[i])))]

    return pd.concat(df_list)






def add_game_date_and_season_to_player_advanced_box_scores_by_season(df_player_advanced_box_scores,
                                                                     df_team_box_scores_2_1,
                                                                     season_list):
    
    df_team_box_scores_2_1_2009_10_2017_18 = \
    filter_data_frame_by_season(df=df_team_box_scores_2_1,
                                season_list=season_list)

    df_player_advanced_box_scores_4_2_2009_10_2017_18 = \
    pd.merge(df_team_box_scores_2_1_2009_10_2017_18.loc[:, ['game_id', 'game_date', 'team_id','season']].rename(columns={'game_id':'GAME_ID', 'team_id':'TEAM_ID'}),
             df_player_advanced_box_scores,
             how='inner',
             on=['GAME_ID', 'TEAM_ID'])
    
    sr(df_player_advanced_box_scores_4_2_2009_10_2017_18, ['game_date', 'GAME_ID'])
    
    return df_player_advanced_box_scores_4_2_2009_10_2017_18





def clean_player_advanced_box_scores_4_2(df_player_advanced_box_scores_4_2_2009_10_2017_18):

    def fill_numeric_columns_with_zero(df_player_advanced_box_scores_4_2_2009_10_2017_18,
                                       columns_excluded_list=['GAME_ID', 'TEAM_ID', 'PLAYER_ID']):

        column_name_list_numeric = \
        list(df_player_advanced_box_scores_4_2_2009_10_2017_18.select_dtypes('number').columns)

        column_name_list_numeric_not_game_id_team_id_player_id = \
        [k for k in column_name_list_numeric if not k in columns_excluded_list]

        df_player_advanced_box_scores_4_2_2009_10_2017_18.loc[:, column_name_list_numeric_not_game_id_team_id_player_id] = \
        df_player_advanced_box_scores_4_2_2009_10_2017_18.loc[:, column_name_list_numeric_not_game_id_team_id_player_id].fillna(0)

        return df_player_advanced_box_scores_4_2_2009_10_2017_18



    def change_MIN_object_to_float64(df_player_advanced_box_scores_4_2_2009_10_2017_18):

        df_player_advanced_box_scores_4_2_2009_10_2017_18.loc[:, 'MIN'] = \
        df_player_advanced_box_scores_4_2_2009_10_2017_18.loc[:, 'MIN'].fillna('00:00')

        df_player_advanced_box_scores_4_2_2009_10_2017_18.loc[:, 'MIN'] = \
        pd.to_timedelta('00:' + df_player_advanced_box_scores_4_2_2009_10_2017_18.loc[:, 'MIN']).astype('timedelta64[s]').astype('float64') / 60

        return df_player_advanced_box_scores_4_2_2009_10_2017_18
    
    #fill null Player Stats with zero and fix data type(s)
    df_player_advanced_box_scores_4_2_2009_10_2017_18 = \
    fill_numeric_columns_with_zero(df_player_advanced_box_scores_4_2_2009_10_2017_18,
                                       columns_excluded_list=['GAME_ID', 'TEAM_ID', 'PLAYER_ID'])


    df_player_advanced_box_scores_4_2_2009_10_2017_18 = \
    change_MIN_object_to_float64(df_player_advanced_box_scores_4_2_2009_10_2017_18)
    
    
    return df_player_advanced_box_scores_4_2_2009_10_2017_18






def filter_player_injury_report_by_game_date(df_player_injury_report,
                                      df_team_box_scores_2_1_away_home,
                                      season_list):

    game_date_min = \
    df_team_box_scores_2_1_away_home.loc[df_team_box_scores_2_1_away_home.loc[:,'season'].isin([season_list[0]])].game_date.min()

    game_date_max = \
    df_team_box_scores_2_1_away_home.loc[df_team_box_scores_2_1_away_home.loc[:,'season'].isin([season_list[1]])].game_date.max()


    df_player_injury_report = \
    df_player_injury_report.loc[(df_player_injury_report.Date >= game_date_min) &
                         (df_player_injury_report.Date <= game_date_max),
                         :]
    
    return df_player_injury_report.sort_values(['Date']).reset_index(drop=True)


#this going to be used??
def filter_player_injury_report_by_team(df_player_injury_report):
    
    df_player_injury_report = \
    df_player_injury_report.loc[df_player_injury_report.loc[:, 'Team'].notnull(), :]
    
    df_player_injury_report = \
    df_player_injury_report.loc[df_player_injury_report.loc[:, 'Team'] != 'Bullets', :]
    
    df_player_injury_report = \
    df_player_injury_report.reset_index(drop=True)
    
    return df_player_injury_report




def get_team_id_abbreviation_nickname(df_team_id_abbreviation_nickname,
                                      other,
                                      df_player_injury_report=None):
    '''
    get team_id, abbreviation, and nickname by the injury report, team name-abbreviation dataset, and other.'''
    
    
    df_team_id_abbreviation_nickname = \
    df_team_id_abbreviation_nickname.loc[:, ['id','abbreviation', 'nickname']].replace({'CHH':'CHA', 
                                                                                        'NJN':'BKN', 
                                                                                        'SEA':'OKC', 
                                                                                        'VAN':'MEM'})
    #NJN is not wrong here...
    
    #create Charlotte Bobcats data frame.
    df_bobcats = df_team_id_abbreviation_nickname.iloc[29].to_frame().T
    df_bobcats.loc[29, 'nickname'] = 'Bobcats'
    
    #create New Orlean Hornets data frame.
    df_hornets = df_team_id_abbreviation_nickname.iloc[3,:].to_frame().T
    df_hornets.loc[:, 'abbreviation'] = 'NOH'
    df_hornets.loc[:, 'nickname'] = 'Hornets'

    
    #add Charlotte Bobcats and New Orlean Hornets data frames to df_team_id_abbreviation_nickname_
    df_team_id_abbreviation_nickname = \
    pd.concat([df_team_id_abbreviation_nickname, 
               df_bobcats,
               df_hornets])

    df_team_id_abbreviation_nickname = \
    df_team_id_abbreviation_nickname.reset_index(drop=True)
    
    
    #https://www.reddit.com/r/nba/comments/6xw0li/can_someone_explain_the_whole_bobcats_hornets/
    '''
    Charlotte Hornets were a team. They then moved to New Orleans, becoming the New Orleans Hornets. Charlotte
    got a new team named the Charlotte Bobcats, as there was already a Hornets team. When Tom Benson bought the
    New Orleans Hornets he decided to change the name to Pelicans. This allowed Charlotte, still fond of their
    Hornets name that was originally theirs, to drop the Bobcats name and become the Charlotte Hornets again.
    '''
    
    return df_team_id_abbreviation_nickname.rename(columns={'id':'team_id'})



def get_first_and_last_game_date_per_season(df_team_box_scores_2_1_away_home,
                                            season_list):
    first = []
    last = []
    
    for i in range(len(season_list)):
        first.append(df_team_box_scores_2_1_away_home.loc[df_team_box_scores_2_1_away_home.loc[:, 'season'] == season_list[i],
                                                      'game_date'].min())
        
        last.append(df_team_box_scores_2_1_away_home.loc[df_team_box_scores_2_1_away_home.loc[:, 'season'] == season_list[i],
                                                     'game_date'].max())
    
    df_season_first_game_date_last_game_date = \
    pd.DataFrame({'season': season_list, 'first': first, 'last':last})
    
    return df_season_first_game_date_last_game_date
    

def get_and_clean_player_inactives_data_frame():
    #read the inactive player dataset into a data frame.
    path = os.path.join('..', '02_raw_data','basketball.sqlite')
    con = sqlite3.connect(path)

    df_player_inactives = pd.read_sql_query("SELECT * from Game_Inactive_Players", con)

    #add 'player' column.
    df_player_inactives.loc[:, 'player'] = \
    df_player_inactives.loc[:, 'FIRST_NAME'] + " " + df_player_inactives.loc[:, 'LAST_NAME']

    df_player_inactives.loc[:, 'player'] = \
    df_player_inactives.loc[:, 'player'].str.strip()

    #choose data type
    df_player_inactives.loc[:, 'GAME_ID'] = \
    df_player_inactives.loc[:, 'GAME_ID'].astype(int)

    df_player_inactives.loc[:, 'TEAM_ID'] = \
    df_player_inactives.loc[:, 'TEAM_ID'].astype(int)
    
    return df_player_inactives




def add_game_date_and_season_and_clean_player_inactives_data_frame(df_player_inactives,
                                                                   df_team_box_scores_2_1_away_home):
    
    def filter_for_2009_10_2017_18_seasons_and_add_game_date_and_season(df_player_inactives,
                                                                        df_team_box_scores_2_1_away_home):
        
        #get game_id, game_date, and season from the box scores dataset for the 2009-10 thru 2017-18 seasons.
        season_list = \
        ['2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18']

        df_team_box_scores_2_1_2009_10_2017_18 = \
        df_team_box_scores_2_1_away_home.loc[df_team_box_scores_2_1_away_home.loc[:, 'season'].isin(season_list), :]

        df_team_box_scores_2_1_2009_10_2017_18 = \
        df_team_box_scores_2_1_2009_10_2017_18.reset_index(drop=True)

        #add game_date and season to inactive players data frame and clean player_id and player columns.
        df_player_inactives_2009_10_2017_18 = \
        pd.merge(df_team_box_scores_2_1_2009_10_2017_18.loc[:, ['game_id', 'game_date', 'season']], 
                 df_player_inactives.rename(columns={'GAME_ID':'game_id'}), 
                 on=['game_id'], 
                 how='inner').reset_index(drop=True)

        df_player_inactives_2009_10_2017_18.loc[:, 'PLAYER_ID'] = \
        df_player_inactives_2009_10_2017_18.loc[:, 'PLAYER_ID'].astype(int)

        return df_player_inactives_2009_10_2017_18
    
    def player_player_id_cleaning(df_player_inactives_2009_10_2017_18):

        #clean player id's and player names.
        def drop_rows(df):
            df = df.loc[df.loc[:, 'PLAYER_ID'] != 1626194, :] #10-day cav contract Marcus Thornton. dob '93. no nba games played

            return df

        def edit_player_id(df):
            df.at[df.loc[df.loc[:, 'PLAYER_ID'] == 27596, :].index.values[0], 'PLAYER_ID'] = 1740 #Rashard Lewis single game PLAYER_ID write

            df.at[df.loc[df.loc[:, 'PLAYER_ID'] == 6274, :].index.values[0], 'PLAYER_ID'] = 1882 #Elton brand single game PLAYER_ID write

            return df

        def replace_player(df):
            df.loc[:, 'player'].replace({'Jianlian Yi':'Yi Jianlian'}, inplace=True)

            df.loc[:, 'player'].replace({'Nene Hilario': 'Nene'}, inplace=True)

            df.loc[:, 'player'].replace({'Ming Yao': 'Yao Ming'}, inplace=True)

            return df 

        df_player_inactives_2009_10_2017_18 = \
        drop_rows(df_player_inactives_2009_10_2017_18).reset_index(drop=True)

        df_player_inactives_2009_10_2017_18 = \
        edit_player_id(df_player_inactives_2009_10_2017_18)

        df_player_inactives_2009_10_2017_18 = \
        replace_player(df_player_inactives_2009_10_2017_18)
        
        return df_player_inactives_2009_10_2017_18
    
    df_player_inactives_2009_10_2017_18 = \
    filter_for_2009_10_2017_18_seasons_and_add_game_date_and_season(df_player_inactives=df_player_inactives,
                                                                    df_team_box_scores_2_1_away_home=df_team_box_scores_2_1_away_home)


    
    return player_player_id_cleaning(df_player_inactives_2009_10_2017_18)









def combine_game_date_game_id_team_id_season_player_id_player_from_data_frames(df_player_advanced_box_scores_2009_10_2017_18,
                                                                           df_player_inactives_2009_10_2017_18):
    
    #get player advanced box scores
    df_player_advanced_box_scores_2009_10_2017_18_game_date_game_id_team_id_season_player_id_player = \
    df_player_advanced_box_scores_2009_10_2017_18.loc[:, ['game_date', 'GAME_ID', 'TEAM_ID', 
                                                          'season', 'PLAYER_ID', 'PLAYER_NAME']].rename(columns={'GAME_ID':'game_id',
                                                                                                                      'TEAM_ID':'team_id',
                                                                                                                      'PLAYER_NAME':'player'}).reset_index(drop=True)
    
    #get inactive player dataset
    df_player_inactives_2009_10_2017_18_game_date_game_id_team_id_season_player_id_player = \
    df_player_inactives_2009_10_2017_18.loc[:, ['game_date', 'game_id', 'TEAM_ID', 
                                                 'season', 'PLAYER_ID', 'player']].rename(columns={'TEAM_ID':'team_id'}).reset_index(drop=True)

    #combine active and inactive player datasets
    df_player_active_2009_10_2017_18_player_inactives_2009_10_2017_18 = \
    pd.concat([df_player_advanced_box_scores_2009_10_2017_18_game_date_game_id_team_id_season_player_id_player, 
               df_player_inactives_2009_10_2017_18_game_date_game_id_team_id_season_player_id_player]).reset_index(drop=True)
    
    #add TEAM_NAME
    df_player_active_2009_10_2017_18_player_inactives_2009_10_2017_18 = \
    pd.merge(df_player_active_2009_10_2017_18_player_inactives_2009_10_2017_18,
             df_player_inactives_2009_10_2017_18.loc[:, ['TEAM_ID', 'season', 'TEAM_NAME']].rename(columns={'TEAM_NAME':'team', 'TEAM_ID':'team_id'}).drop_duplicates(),
             how='inner',
             on=['team_id', 'season'])

    return sr(df_player_active_2009_10_2017_18_player_inactives_2009_10_2017_18, ['game_date', 'game_id'])

















#read in injury report and clean by player, team, and date.
def read_in_player_injury_report_and_clean_player_team_and_date():

    
    def get_player_injury_report_and_extract_player_column():
        df_player_injury_report = rcr('injuries_2010-2020.csv', parse_dates=['Date'])

        df_player_injury_report.loc[:, 'Acquired'] = df_player_injury_report.loc[:, 'Acquired'].fillna('')

        df_player_injury_report.loc[:, 'Relinquished'] = df_player_injury_report.loc[:, 'Relinquished'].fillna('')

        df_player_injury_report.loc[:, 'player'] = df_player_injury_report.loc[:, 'Relinquished'] + ' ' + df_player_injury_report.loc[:, 'Acquired']

        df_player_injury_report.loc[:, 'player'] = df_player_injury_report.loc[:, 'player'].str.strip()

        return df_player_injury_report
    
    def df_inj1020_fix_Date_Team(df):
        df.loc[5923, ['Team']] = 'Pacers'
        return df

    
    def filter_by_last_game_date_of_season(df_player_injury_report,
                                           season='2017-18'):

        df_season_first_game_date_last_game_date = \
        rcp('7season_first_game_date_last_game_date.csv.gz', parse_dates=['first', 'last'])

        last_game_date_of_season = \
        df_season_first_game_date_last_game_date.loc[df_season_first_game_date_last_game_date.loc[:, 'season'].isin([season]), 'last'].values[0]

        df_player_injury_report = \
        df_player_injury_report.loc[df_player_injury_report.loc[:, 'Date'] <= last_game_date_of_season]

        return df_player_injury_report


    def replace_player(df):
        df.loc[:, 'player'].replace({'J.J. Redick':'JJ Redick'}, inplace=True)
        df.loc[:, 'player'].replace({'A.J. Price': 'AJ Price'}, inplace=True)
        df.loc[:, 'player'].replace({'Alex Ajinca':'Alexis Ajinca'}, inplace=True)
        df.loc[:, 'player'].replace({'C.J. Miles': 'CJ Miles'}, inplace=True)
        df.loc[:, 'player'].replace({'D.J. White': 'DJ White'}, inplace=True)
        df.loc[:, 'player'].replace({"Hamady N'Diaye" : 'Hamady Ndiaye'}, inplace=True)
        df.loc[:, 'player'].replace({'J.J. Hickson': 'JJ Hickson'}, inplace=True)
        df.loc[:, 'player'].replace({'John Wallace' : 'John Wall'}, inplace=True)
        df.loc[:, 'player'].replace({'Jose Barea' : 'J.J. Barea'}, inplace=True)
        df.loc[:, 'player'].replace({'Wes Johnson':'Wesley Johnson'}, inplace=True)
        df.loc[:, 'player'].replace({'B.J. Mullens':'Byron Mullens'}, inplace=True)
        df.loc[:, 'player'].replace({'Bill Walker':'Henry Walker'}, inplace=True)
        df.loc[:, 'player'].replace({'Gerald Henderson Jr.':'Gerald Henderson'}, inplace=True)
        df.loc[:, 'player'].replace({'Jianlian Yi':'Yi Jianlian'}, inplace=True)
        df.loc[:, 'player'].replace({'Mike Conley Jr.':'Mike Conley'}, inplace=True)
        df.loc[:, 'player'].replace({'Mike Dunleavy Jr.':'Mike Dunleavy'}, inplace=True)
        df.loc[:, 'player'].replace({'Nene Hilario': 'Nene'}, inplace=True)
        df.loc[:, 'player'].replace({'Didier Mbenga':'DJ Mbenga'}, inplace=True)
        df.loc[:, 'player'].replace({'A.J. Hammons':'AJ Hammons'}, inplace=True)
        df.loc[:, 'player'].replace({'C.J. McCollum':'CJ McCollum'}, inplace=True)
        df.loc[:, 'player'].replace({'D.J. Augustine':'D.J. Augustin'}, inplace=True)
        df.loc[:, 'player'].replace({'DeAndre Bembry':"DeAndre' Bembry"}, inplace=True)
        df.loc[:, 'player'].replace({'Giorgios Papagiannis':'Georgios Papagiannis'}, inplace=True)
        df.loc[:, 'player'].replace({'Glen Rice Jr.':'Glen Rice'}, inplace=True)
        df.loc[:, 'player'].replace({'J.R. Smith':'JR Smith'}, inplace=True)
        df.loc[:, 'player'].replace({'James Ennis':'James Ennis III'}, inplace=True)
        df.loc[:, 'player'].replace({'John Collins  John Collins':'John Collins'}, inplace=True)
        df.loc[:, 'player'].replace({'Juan Hernangomez':'Juancho Hernangomez'}, inplace=True)
        df.loc[:, 'player'].replace({'Billy Ouattara':'Yakuba Ouattara'}, inplace=True)
        df.loc[:, 'player'].replace({'Courtney Simpson':'Diamon Simpson'}, inplace=True)
        df.loc[:, 'player'].replace({'Danuel House':'Danuel House Jr.'}, inplace=True)
        df.loc[:, 'player'].replace({'Domas Sabonis':'Domantas Sabonis'}, inplace=True)
        df.loc[:, 'player'].replace({'Jake Wiley':'Jacob Wiley'}, inplace=True)
        df.loc[:, 'player'].replace({'James McAdoo':'James Michael McAdoo'}, inplace=True)
        df.loc[:, 'player'].replace({'Jeff Pendergraph':'Jeff Ayres'}, inplace=True)
        df.loc[:, 'player'].replace({'Jeff Taylor':'Jeffery Taylor'}, inplace=True)
        df.loc[:, 'player'].replace({'K.J. McDaniels':'KJ McDaniels'}, inplace=True)
        df.loc[:, 'player'].replace({'K.J. McDaniels':'KJ McDaniels'}, inplace=True)
        df.loc[:, 'player'].replace({'Malcom Lee':'Malcolm Lee'}, inplace=True)
        df.loc[:, 'player'].replace({'Marcus Morris':'Marcus Morris Sr.'}, inplace=True)
        df.loc[:, 'player'].replace({'Matt Williams':'Matt Williams Jr.'}, inplace=True)
        df.loc[:, 'player'].replace({"Maurice N'dour":'Maurice Ndour'}, inplace=True)
        df.loc[:, 'player'].replace({'Milos Tedosic':'Milos Teodosic'}, inplace=True)
        df.loc[:, 'player'].replace({'Moe Harkless':'Maurice Harkless'}, inplace=True)
        df.loc[:, 'player'].replace({'O.G. Anunoby':'OG Anunoby'}, inplace=True)
        df.loc[:, 'player'].replace({'Ognen Kuzmic':'Ognjen Kuzmic'}, inplace=True)
        df.loc[:, 'player'].replace({'P.J. Dozier':'PJ Dozier'}, inplace=True)
        df.loc[:, 'player'].replace({'P.J. Hairston':'PJ Hairston'}, inplace=True)
        df.loc[:, 'player'].replace({'Qi Zhou':'Zhou Qi'}, inplace=True)
        df.loc[:, 'player'].replace({'R.J. Hunter':'RJ Hunter'}, inplace=True)
        df.loc[:, 'player'].replace({'Ray McCallum Jr.':'Ray McCallum'}, inplace=True)
        df.loc[:, 'player'].replace({'Ron Artest':'Metta World Peace'}, inplace=True)
        df.loc[:, 'player'].replace({'Roy Marble':'Devyn Marble'}, inplace=True)
        df.loc[:, 'player'].replace({'Stephen Zimmerman Jr.':'Stephen Zimmerman'}, inplace=True)
        df.loc[:, 'player'].replace({'Tony Wroten Jr.':'Tony Wroten'}, inplace=True)
        df.loc[:, 'player'].replace({'Toure Murry':"Toure' Murry"}, inplace=True)
        df.loc[:, 'player'].replace({'Viacheslav Kravstov':'Viacheslav Kravtsov'}, inplace=True)
        df.loc[:, 'player'].replace({'Vince Hunter':'Vincent Hunter'}, inplace=True)
        df.loc[:, 'player'].replace({'Walker Russell Jr.':'Walker Russell'}, inplace=True)
        df.loc[:, 'player'].replace({'Wayne Selden Jr.':'Wayne Selden'}, inplace=True)
        df.loc[:, 'player'].replace({'Wes Matthews Jr.':'Wesley Matthews'}, inplace=True)
        return df



    #names dropped from Injury Report dataframe df_inj1018 for the following reasons:
    #a. no player nba regular season or playoff boxscore advanced stats between the 2009-10 and 2017-18 seasons.
    #b. player did not play an nba regular season or playoff game between the 2009-10 and 2017-18 seasons.
    #c. no player nba regular season or playoff boxscore advanced stats between the 2009-10 and 2017-18 seasons. Player plays in nba games
    #   following the 2017-18 season.
    #d. this is the injury of a coach, not a player. coaches do not have boxscore advanced stats


    def drop_player(df):
        df.drop(df.loc[df.loc[:, 'player'].str.contains('Marion Hillard'), :].index, inplace = True) #a
        df.drop(df.loc[df.loc[:, 'player'].str.contains('Matt Janning'), :].index, inplace = True) #a
        df.drop(df.loc[df.loc[:, 'player'].str.contains("Da'Sean Butler"), :].index, inplace = True) #a
        df.drop(df.loc[df.loc[:, 'player'].str.contains('Eric Griffin'), :].index, inplace = True) #a
        df.drop(df.loc[df.loc[:, 'player'].str.contains('Harry Giles'), :].index, inplace = True) #c
        df.drop(df.loc[df.loc[:, 'player'].str.contains('Magnum Rolle'), :].index, inplace = True) #a
        df.drop(df.loc[df.loc[:, 'player'].str.contains('Robert Vaden'), :].index, inplace = True) #a
        df.drop(df.loc[df.loc[:, 'player'].str.contains('Steve Clifford'), :].index, inplace = True) #d
        df.drop(df.loc[df.loc[:, 'player'].str.contains('Steve Kerr'), :].index, inplace = True) #d
        df.drop(df.loc[df.loc[:, 'player'].str.contains('Terrico White'), :].index, inplace = True)#a
        df.drop(df.loc[df.loc[:, 'player'].str.contains('Tyronn Lue'), :].index, inplace = True) #d
        return df

    
    
    def fix_chris_wright_Team(df):
        indexes = list(df.loc[((df.loc[:, 'Date'] == '2011-12-25') |
                               (df.loc[:, 'Date'] == '2011-12-28') |
                               (df.loc[:, 'Date'] == '2012-01-04') |
                               (df.loc[:, 'Date'] == '2012-01-20'))
                              & (df.loc[:, 'player'] == 'Chris Wright'), :].index)

        df.at[indexes, 'Team'] = 'Warriors'
        
        return df

    
    
    #drop '2011-12-25', 'Jeff Green' from df_player_injury_report bc he didn't play the 2011-12 season
    def drop_jeff_green_2011_12(df):
        indexes = list(df.loc[((df.loc[:, 'Date'] == '2011-12-17') | 
                               (df.loc[:, 'Date'] == '2011-12-25'))
                              & (df.loc[:, 'player'] == 'Jeff Green'), :].index)
        
        df.drop(indexes, inplace=True)
        
        return df
    #not seen as inactive/active on nba.com boxscores for the 2011-12 season

    #'preseason physical with his employer, the Boston Celtics...failed his physical, voiding his one-year deal with the team, and won't play at all in the 2011-12 season'
    #https://www.espn.com/nba/story/_/id/23713648/nba-how-jeff-green-returned-heart-defect-almost-ended-life
    
    
    

    #fix Team from Cavaliers to Pistons on '2014-04-04' and '2013-11-29' game_date's for 'Will Bynum'
    def fix_will_bynum_team_2013_14(df):

        indexes = list(df.loc[df.loc[:, 'player'].str.contains('Will Bynum') & 
                              ((df.loc[:, 'Date'] == '2014-04-04') | 
                               (df.loc[:, 'Date'] == '2013-11-29')), :].index)

        df.at[indexes, 'Team'] = 'Pistons'

        return df
    
    
    #drop jason kidd for game_date '2015-12-20' since he was a coach here
    def drop_jason_kidd_2015_16(df):
        index = list(df.loc[df.loc[:, 'player'].str.contains('Jason Kidd') & 
                            (df.loc[:, 'Date'] == '2015-12-20'), :].index)
        
        df.drop(index, inplace=True)
        
        return df

    
    #fix Team from hornets to Cavaliers for Chris Andersen on game_date '2017-02-13'
    def fix_chris_andersen_Team(df):
        index = list(df.loc[df.loc[:, 'player'].str.contains('Chris Andersen') & 
                            (df.loc[:, 'Date'] == '2017-02-13'), :].index)
        
        df.at[index, 'Team']  = 'Cavaliers'
        
        return df
    


    #Bogdan Bogdanovic on game_date '2017-10-18' should have Team changed from Pacers to Kings since the Pacers play has a first name of Bojan not Bogdan
    def fix_bogdan_bogdanovic_Team(df):
        index = list(df.loc[df.loc[:, 'player'].str.contains('Bogdan Bogdanovic') & 
                            (df.loc[:, 'Date'] == '2017-10-18'), :].index)
        
        df.at[index, 'Team'] = 'Kings'
        
        return df
    

    #for Date '2017-02-13' and player/Relinquished 'Chris Wilcox', replace name with C.J. Wilcox and Team with 'Magic'

    #earlier in season tenditis, but not on specific date https://www.nba.com/clippers/news/hand-injury-frustrating-wilcox-after-finally-feeling-healthy
    #Wilcox of Magic inactive https://statsdmz.nba.com/pdfs/20170213/20170213_ORLMIA.pdf
    #C.J. Wilcox of the Magic that 2016-17 season https://www.basketball-reference.com/players/w/wilcocj01.html
    def fix_cj_wilcox_Team_player(df):
        index = list(df.loc[(df.loc[:, 'Date'] == '2017-02-13') & 
                            df.loc[:, 'player'].str.contains('Chris Wilcox'), :].index)
        
        df.at[index, 'Team'] = 'Magic'
        
        df.at[index, 'player'] = 'C.J. Wilcox'
        
        if True in df.columns.str.contains('Relinquished'):
            df.at[index, 'Relinquished'] = 'C.J. Wilcox'
        
        return df

    def drop_okaro_white_sheldon_mac_hawks_2018_02_08(df):
        index = list(df.loc[df.loc[:, 'player'].str.contains('Okaro White') & 
                              (df.loc[:, 'Date'] == '2018-02-08'), :].index)
        
        df.drop(index, inplace=True)

        
        index = list(df.loc[df.loc[:, 'player'].str.contains('Sheldon Mac') & 
                              (df.loc[:, 'Date'] == '2018-02-08'), :].index)
        
        df.drop(index, inplace=True)

        return df

    #Andrew White of the hawks was an inactive player on this Team and date. why does the injury report show player was Acquired???
    #andrew white inactive https://statsdmz.nba.com/pdfs/20180208/20180208_ATLORL.pdf
    #andrew white, hawk the 2017-18 season https://www.basketball-reference.com/players/w/whitean01.html


    #okaro white , heat, left foot surgery inactive https://statsdmz.nba.com/pdfs/20180207/20180207_HOUMIA.pdf
    # okaro white, not on heat inactive list the following nba game https://statsdmz.nba.com/pdfs/20180209/20180209_MILMIA.pdf

    #Sheldon Mac
    #'He did not appear in a game for the Wizards in 2017–18, and on February 8, 2018,
    # he was traded to the Atlanta Hawks alongside cash considerations in exchange for a protected 2019 second round draft pick.'


    
    
    
    #get injury report and extract 'player'
    df_player_injury_report = \
    get_player_injury_report_and_extract_player_column()


    #fix a single entry Team
    df_player_injury_report = \
    df_inj1020_fix_Date_Team(df_player_injury_report)

    
    #filter for the last game in the 2017-18 season
    df_player_injury_report = \
    filter_by_last_game_date_of_season(df_player_injury_report=df_player_injury_report,
                                       season='2017-18')

    #fix many player (name) entries
    df_player_injury_report = \
    replace_player(df_player_injury_report)

    #drop a handful of rows by player (name)
    df_player_injury_report = \
    drop_player(df_player_injury_report)


    #fix chris wright team to 'Warriors' on 4 dates
    df_player_injury_report = \
    fix_chris_wright_Team(df_player_injury_report)

    

    #drop 2011-12 season rows for jeff green since he was out for the season to heart surgery. contract voided.
    df_player_injury_report = \
    drop_jeff_green_2011_12(df_player_injury_report)


    #fix Team (name) for Will Bynum on 2 dates
    df_player_injury_report = \
    fix_will_bynum_team_2013_14(df_player_injury_report)



    #drop jason kidd for game_date '2015-12-20' since he was a coach here
    df_player_injury_report = \
    drop_jason_kidd_2015_16(df_player_injury_report)



    #fix Team from hornets to Cavaliers for Chris Andersen on game_date '2017-02-13'
    df_player_injury_report = \
    fix_chris_andersen_Team(df_player_injury_report)




    #Bogdan Bogdanovic on game_date '2017-10-18' should have Team changed from Pacers to Kings since the Pacers play has a first name of Bojan not Bogdan
    df_player_injury_report = \
    fix_bogdan_bogdanovic_Team(df_player_injury_report)


    df_player_injury_report = \
    fix_cj_wilcox_Team_player(df_player_injury_report)



    #okaro white get his Team, Notes, Acquired, Relinquished, Date messed with?????
    df_player_injury_report = \
    drop_okaro_white_sheldon_mac_hawks_2018_02_08(df_player_injury_report)
    
    
    #change 'Blazers' to 'Trail Blazers'
    df_player_injury_report.loc[:, 'Team'] = \
    df_player_injury_report.loc[:, 'Team'].replace({'Blazers':'Trail Blazers'})

    
    return df_player_injury_report








def add_team_id_season_game_id_PLAYER_ID_filter_for_Relinquished_on_team_game_date_and_drop_columns(df_player_injury_report,
                                                                                                    df_player_active_2009_10_2017_18_player_inactives_2009_10_2017_18,
                                                                                                    drop_columns=['Acquired', 'Relinquished', 'Notes']):
    
    #add 'team_id', 'season', 'game_id', and 'PLAYER_ID'; Filter for player injury reported on a team 'game_date'.

    #add 'team_id', 'season', and 'game_id' and Filter for player injury reported on a team 'game_date'.
    df_player_injury_report = \
    pd.merge(df_player_injury_report.rename(columns={'Team':'team', 'Date':'game_date'}),
             df_player_active_2009_10_2017_18_player_inactives_2009_10_2017_18.loc[:, ['game_date', 'team', 'team_id', 'season', 'game_id']].drop_duplicates(),
             how='inner',
             on=['game_date', 'team'])

    #add 'PLAYER_ID'

    df_player_injury_report = \
    pd.merge(df_player_injury_report,
             df_player_active_2009_10_2017_18_player_inactives_2009_10_2017_18.loc[:, ['team_id', 'season', 'player', 'PLAYER_ID']].drop_duplicates(),
             how='inner',
             on=['team_id', 'season', 'player'])


    
    #filter for player 'Relinquished' rows. Drop 'Acquired', 'Relinquished', 'Notes' columns
    df_player_injury_report = \
    df_player_injury_report.loc[df_player_injury_report.loc[:, 'Relinquished'] != '', :].drop(columns=drop_columns).drop_duplicates()


    #sort columns and rows
    column_name_list = \
    ['game_date', 'game_id', 'team', 'team_id', 'player', 'PLAYER_ID', 'season']


    df_player_injury_report = \
    df_player_injury_report.loc[:, column_name_list].sort_values(['game_date', 'game_id', 'team_id', 'PLAYER_ID']).reset_index(drop=True)
    
    return df_player_injury_report



def print_interseason_team_name_changes(df_player_injury_report):
    '''
    print team (name) changes by interseason.'''


    #get team (name) and season
    df_team_team_id_season = \
    df_player_injury_report.loc[:, ['team', 'team_id', 'season']].drop_duplicates().reset_index(drop=True)


    #get season transitions
    season_list = \
    df_player_injury_report.loc[:, 'season'].drop_duplicates().to_list()

    df_season_former_latter = \
    pd.DataFrame({'former': season_list[0:7], 'latter':season_list[1:8]})



    for row in range(7):
        former_season = df_season_former_latter.loc[row, 'former']
        latter_season = df_season_former_latter.loc[row, 'latter']


        former_team_list = \
        list(set(df_team_team_id_season.loc[df_team_team_id_season.loc[:, 'season'] == former_season, 'team']) - \
             set(df_team_team_id_season.loc[df_team_team_id_season.loc[:, 'season'] == latter_season, 'team']))

        latter_team_list = \
        list(set(df_team_team_id_season.loc[df_team_team_id_season.loc[:, 'season'] == latter_season, 'team']) - \
             set(df_team_team_id_season.loc[df_team_team_id_season.loc[:, 'season'] == former_season, 'team']))


        if (len(former_team_list) > 0) & (len(latter_team_list) > 0):
            print(former_season + ' and ' + latter_season + ' ' +
                  str(former_team_list) + '-->' + str(latter_team_list))




            
            


            
def get_and_clean_team_advanced_box_scores():
    def get_and_combine_team_advanced_box_score_splits():
        '''
        get and combine team advanced box scores.'''


        #get team advanced box score filename splits
        filename_list_team_advanced_stats = os.listdir(os.path.join('..', '04_processed_data_preliminary'))
        filename_list_team_advanced_stats.remove('.DS_Store')

        filename_list_team_advanced_stats = [k for k in filename_list_team_advanced_stats if ('5game_id_team_id_advanced_stats' in k) & ('.csv' in k)]


        #read in team advanced box score filename splits and combine 
        df_team_advanced_box_scores = pd.DataFrame({})

        for i in range(len(filename_list_team_advanced_stats)):
            df_temp = pd.read_csv(os.path.join('..', '04_processed_data_preliminary', str(filename_list_team_advanced_stats[i])))
            df_team_advanced_box_scores = pd.concat([df_team_advanced_box_scores, df_temp])

        return df_team_advanced_box_scores.reset_index(drop=True)


    def clean_team_advanced_box_scores(df_team_advanced_box_scores):
        #fix LA of 'TEAM_CITY' column to Los Angeles
        df_team_advanced_box_scores.loc[df_team_advanced_box_scores.loc[:, 'TEAM_CITY'] == 'LA', 'TEAM_CITY'] = 'Los Angeles'
        

        return df_team_advanced_box_scores


    df_team_advanced_box_scores = \
    get_and_combine_team_advanced_box_score_splits()



    df_team_advanced_box_scores = \
    clean_team_advanced_box_scores(df_team_advanced_box_scores)
    
    #clean Team Advanced Box Scores of Team Advanced Box Scores away home and Team Box Scores away home.

    return df_team_advanced_box_scores







def extract_duration_and_duration_minutes_columns(df_team_advanced_box_scores):
    #'MIN' string to 'duration' timedelta

    def extract_hours_minutes_seconds(df_team_advanced_box_scores,
                                      column_name='MIN'):
        '''
        take extract hours, minutes, and seconds from 'MIN' column.'''

        df_hours_minutes_seconds = \
        df_team_advanced_box_scores.loc[:, column_name].str.split(':', expand=True).rename(columns={0:'minutes', 1:'seconds'})

        df_hours_minutes_seconds.loc[:, 'total_seconds'] = \
        df_hours_minutes_seconds.loc[:, 'minutes'].astype('int64') * 60 + df_hours_minutes_seconds.loc[:, 'seconds'].astype('int64').values

        df_hours_minutes_seconds.loc[:, 'hours'] = \
        np.floor(df_hours_minutes_seconds.loc[:, 'total_seconds'] / 3600).astype('int64')

        df_hours_minutes_seconds.loc[:, 'seconds_remaining_hours_subtracted'] = \
        df_hours_minutes_seconds.loc[:, 'total_seconds'] - df_hours_minutes_seconds.loc[:, 'hours'] * 3600

        df_hours_minutes_seconds.loc[:, 'minutes'] = \
        (np.floor(df_hours_minutes_seconds.loc[:, 'seconds_remaining_hours_subtracted'] / 60)).astype('int64')

        df_hours_minutes_seconds.loc[:, 'seconds_remaining_hours_minutes_subtracted'] = \
        (df_hours_minutes_seconds.loc[:, 'seconds_remaining_hours_subtracted'] - df_hours_minutes_seconds.loc[:, 'minutes'] * 60).astype('int64')

        return df_hours_minutes_seconds


    def convert_hours_minutes_seconds_to_string(df_hours_minutes_seconds):
        #convert hours, minutes, and seconds columns to string_hours, string_minutes, string_seconds.
        #add a 0 to the front of single digit strings.

        #convert 'hours' column to 2-digit 'string_hours' column
        df_hours_minutes_seconds.loc[:, 'string_hours'] = \
        df_hours_minutes_seconds.loc[:, 'hours'].astype('string')

        df_hours_minutes_seconds.loc[df_hours_minutes_seconds.loc[:, 'string_hours'].apply(len) == 1, 'string_hours'] = \
        '0' + df_hours_minutes_seconds.loc[df_hours_minutes_seconds.loc[:, 'string_hours'].apply(len) == 1, 'string_hours']

        #convert 'minutes' column to 'string_hours' column
        df_hours_minutes_seconds.loc[:, 'string_minutes'] = \
        df_hours_minutes_seconds.loc[:, 'minutes'].astype('string')

        df_hours_minutes_seconds.loc[df_hours_minutes_seconds.loc[:, 'string_minutes'].apply(len) == 1, 'string_minutes'] = \
        '0' + df_hours_minutes_seconds.loc[df_hours_minutes_seconds.loc[:, 'string_minutes'].apply(len) == 1, 'string_minutes']

        #convert remaining seconds column to 'string_seconds' column
        df_hours_minutes_seconds.loc[:, 'string_seconds'] = \
        df_hours_minutes_seconds.loc[:, 'seconds_remaining_hours_minutes_subtracted'].astype('string')

        df_hours_minutes_seconds.loc[df_hours_minutes_seconds.loc[:, 'string_seconds'].apply(len) == 1, 'string_seconds'] = \
        '0' + df_hours_minutes_seconds.loc[df_hours_minutes_seconds.loc[:, 'string_seconds'].apply(len) == 1, 'string_seconds']

        return df_hours_minutes_seconds


    def extract_timedelta_duration_column_from_columns(df_hours_minutes_seconds,
                                                       column_name_list=['string_hours', 'string_minutes', 'string_seconds']):
        #extract timedelta 'duration' column
        df_hours_minutes_seconds.loc[:, 'duration'] = \
        df_hours_minutes_seconds.loc[:, column_name_list[0]] + ':' + df_hours_minutes_seconds.loc[:, column_name_list[1]] + ':' + df_hours_minutes_seconds.loc[:, column_name_list[2]]

        df_hours_minutes_seconds.loc[:, 'duration'] = pd.to_timedelta(df_hours_minutes_seconds.loc[:, 'duration'])

        return df_hours_minutes_seconds


    df_hours_minutes_seconds = \
    extract_hours_minutes_seconds(df_team_advanced_box_scores=df_team_advanced_box_scores,
                                  column_name='MIN')

    df_hours_minutes_seconds = \
    convert_hours_minutes_seconds_to_string(df_hours_minutes_seconds)

    df_hours_minutes_seconds = \
    extract_timedelta_duration_column_from_columns(df_hours_minutes_seconds,
                                                   column_name_list=['string_hours', 'string_minutes', 'string_seconds'])



    #add team game timedelta 'duration' column to df_team_advanced_box_scores
    df_team_advanced_box_scores = \
    pd.concat([df_team_advanced_box_scores, df_hours_minutes_seconds.loc[:, 'duration']],
              axis=1)

    #extract team game duration in minutes 'duration_minutes' column
    df_team_advanced_box_scores.loc[:, 'duration_minutes'] = \
    df_team_advanced_box_scores.loc[:, 'duration'].dt.seconds / 60

    return df_team_advanced_box_scores














def convert_team_advanced_box_scores_to_away_home(df_team_advanced_box_scores,
                                                  df_team_box_scores_2_1_away_home_2010_11_2017_18):
    
    def add_game_date_team_id_away_and_team_id_away_columns_to_team_advanced_box_scores(df_team_advanced_box_scores,
                                                                                        df_team_box_scores_2_1_away_home_2010_11_2017_18):
        '''
        add game_date, team_id_away, and team_id_home columns.'''
        df_team_advanced_box_scores = \
        pd.merge(df_team_box_scores_2_1_away_home_2010_11_2017_18.loc[:, ['game_id', 'game_date', 'team_id_away', 'team_id_home']], 
                 df_team_advanced_box_scores.rename(columns={'GAME_ID':'game_id'}),
                 on=['game_id'], 
                 how='inner')
        
        return df_team_advanced_box_scores    
    
    def get_and_rename_away_team_advanced_box_scores(df_team_advanced_box_scores):

        #filter for away team rows: df_team_advanced_box_scores_away
        df_team_advanced_box_scores_away = \
        df_team_advanced_box_scores.loc[df_team_advanced_box_scores.loc[:, 'TEAM_ID'] == df_team_advanced_box_scores.loc[:, 'team_id_away'], :]\
        .drop(columns=['team_id_away', 'team_id_home'])

        df_team_advanced_box_scores_away = \
        df_team_advanced_box_scores_away.sort_values(['game_date', 'game_id']).reset_index(drop=True)


        #rename team advanced box score columns to include '_away' suffix
        column_name_list = \
        [k for k in df_team_advanced_box_scores_away.columns if k not in ['game_id', 'game_date']]

        column_name_list_away = \
        [k + '_away' for k in column_name_list]

        column_name_dict_away = \
        dict(zip(column_name_list, column_name_list_away))

        df_team_advanced_box_scores_away = \
        df_team_advanced_box_scores_away.rename(columns=column_name_dict_away)
        
        return df_team_advanced_box_scores_away, column_name_list


    def get_and_rename_home_team_advanced_box_scores(df_team_advanced_box_scores,
                                                     column_name_list):
        
        #filter for home team rows: df_team_advanced_box_scores_home
        df_team_advanced_box_scores_home = \
        df_team_advanced_box_scores.loc[df_team_advanced_box_scores.loc[:, 'TEAM_ID'] == df_team_advanced_box_scores.loc[:, 'team_id_home'], :]\
        .drop(columns=['team_id_away', 'team_id_home'])

        df_team_advanced_box_scores_home = \
        df_team_advanced_box_scores_home.sort_values(['game_date', 'game_id']).reset_index(drop=True)



        #rename team advanced box score columns to include '_home' suffix
        column_name_list_home = \
        [k + '_home' for k in column_name_list]

        column_name_dict_home = \
        dict(zip(column_name_list, column_name_list_home))

        df_team_advanced_box_scores_home = \
        df_team_advanced_box_scores_home.rename(columns=column_name_dict_home)
        
        return df_team_advanced_box_scores_home
    
    
    df_team_advanced_box_scores = \
    add_game_date_team_id_away_and_team_id_away_columns_to_team_advanced_box_scores(df_team_advanced_box_scores,
                                                                                    df_team_box_scores_2_1_away_home_2010_11_2017_18)

    
    df_team_advanced_box_scores_away, column_name_list = \
    get_and_rename_away_team_advanced_box_scores(df_team_advanced_box_scores)


    df_team_advanced_box_scores_home = \
    get_and_rename_home_team_advanced_box_scores(df_team_advanced_box_scores,
                                                 column_name_list)


    #combine away and home team advanced box score data frames.
    df_team_advanced_box_scores_away_home = \
    pd.merge(df_team_advanced_box_scores_away,
             df_team_advanced_box_scores_home,
             on=['game_id', 'game_date'],
             how='inner')

    return df_team_advanced_box_scores_away_home.sort_values(['game_date', 'game_id']).reset_index(drop=True)



def extract_and_add_matchup_column(df_team_advanced_box_scores_away_home):
    #extract matchup column

    #combine away and home team 3-letter abbreviations as a 6-letter string
    df_team_advanced_box_scores_away_home.loc[:, 'matchup'] = \
    df_team_advanced_box_scores_away_home.loc[:, 'TEAM_ABBREVIATION_away'] + df_team_advanced_box_scores_away_home.loc[:, 'TEAM_ABBREVIATION_home']



    #alphabetically order team abbreviations per matchup
    matchup_list = df_team_advanced_box_scores_away_home.loc[:, 'matchup'].drop_duplicates().to_list()

    reverse_matchup_dict = {}
    for i, matchup in enumerate(matchup_list):
        if (matchup[0:3] > matchup[3:6]):
            reverse_matchup_dict[matchup[0:3] + matchup[3:6]] = matchup[3:6] + matchup[0:3]


    df_team_advanced_box_scores_away_home.loc[:, 'matchup'] = \
    df_team_advanced_box_scores_away_home.loc[:, 'matchup'].replace(reverse_matchup_dict)




    #add ' vs. ' to 'matchup' column entries
    matchup_list_alphabetical = \
    list(df_team_advanced_box_scores_away_home.loc[:, 'matchup'].drop_duplicates())

    matchup_dict_alphabetical = {}
    for matchup in matchup_list_alphabetical:
        matchup_dict_alphabetical[matchup] = matchup[0:3] + ' vs. ' + matchup[3:6]


    df_team_advanced_box_scores_away_home.loc[:, 'matchup'] = \
    df_team_advanced_box_scores_away_home.loc[:, 'matchup'].replace(matchup_dict_alphabetical)
    
    return df_team_advanced_box_scores_away_home




#clean Team Advanced Box Scores Offensive Rebound Percentage, Defensive Rebound Percentage, and Rebound Percentage for Away and Home

def fill_missing_team_offensive_rebound_percentage_defensive_rebound_percentage_and_rebound_percentage_for_away_and_home(
    df_team_advanced_box_scores_away_home, 
    df_team_box_scores_away_home_2010_11_2017_18):
    
    '''
    Fill missing Team Advanced Box Score Offensive Rebound Percentage, Defensive, Rebound Percentage, and Rebound Percentage for away and home.'''
    
    #use a df_team_box_scores_away_home_2010_11_2017_18 copy instead of reference
    df_team_box_scores_away_home_2010_11_2017_18 = df_team_box_scores_away_home_2010_11_2017_18.copy()
    
    #get column name list to be calculated from Team Box Scores away homee
    column_name_list_oreb_pct_dreb_pct_reb_pct_away_home = \
    ['oreb_pct_away', 'dreb_pct_away', 'reb_pct', 'oreb_pct_home', 'dreb_pct_home', 'reb_pct_home']
    
    #get column name list to be filled for Team Advanced Box Scores away home
    column_name_list_OREB_PCT_DREB_PCT_REB_PCT_away_home = \
    ['OREB_PCT_away', 'DREB_PCT_away', 'REB_PCT_away', 'OREB_PCT_home', 'DREB_PCT_home', 'REB_PCT_home']
    
    #get column name list for fillna
    column_name_tuple_list_OREB_PCT_DREB_PCT_REB_PCT_away_home = \
    list(zip(column_name_list_OREB_PCT_DREB_PCT_REB_PCT_away_home, column_name_list_oreb_pct_dreb_pct_reb_pct_away_home))
    
    
    #Calculate oreb_pct_away, dreb_pct_away, reb_pct, oreb_pct_home, dreb_pct_home, reb_pct_home from Team Box Scores
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, column_name_list_oreb_pct_dreb_pct_reb_pct_away_home[0]] = \
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'oreb_away'] / (df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'oreb_away'] +
                                                                     df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'dreb_home'])
    
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, column_name_list_oreb_pct_dreb_pct_reb_pct_away_home[1]] = \
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'dreb_away'] / (df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'dreb_away'] +
                                                                     df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'oreb_home'])
    
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, column_name_list_oreb_pct_dreb_pct_reb_pct_away_home[2]] = \
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'reb_away'] / (df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'reb_away'] +
                                                                     df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'reb_home'])
    
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, column_name_list_oreb_pct_dreb_pct_reb_pct_away_home[3]] = \
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'oreb_home'] / (df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'oreb_home'] +
                                                                     df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'dreb_away'])
    
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, column_name_list_oreb_pct_dreb_pct_reb_pct_away_home[4]] = \
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'dreb_home'] / (df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'dreb_home'] +
                                                                     df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'oreb_away'])
    
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, column_name_list_oreb_pct_dreb_pct_reb_pct_away_home[5]] = \
    df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'reb_home'] / (df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'reb_home'] +
                                                                     df_team_box_scores_away_home_2010_11_2017_18.loc[:, 'reb_away'])
    
    #column name list Game ID and Game Date
    column_name_list_merge_on = \
    [column_name for column_name in df_team_advanced_box_scores_away_home.columns if column_name in df_team_box_scores_away_home_2010_11_2017_18.columns]
    
    #column name list Team Box Scores
    column_name_list_oreb_pct_dreb_pct_reb_pct_away_home_game_id_game_date = \
    column_name_list_merge_on + column_name_list_oreb_pct_dreb_pct_reb_pct_away_home
    
    
    
    
    #merge Offensive Rebound Percentage, Defensive Rebound Percentage, and Rebound Percentage away and home to Team Advanced Box Scores
    df_team_advanced_box_scores_away_home_oreb_pct_dreb_pct_reb_pct_away_home = \
    pd.merge(df_team_advanced_box_scores_away_home,
             df_team_box_scores_away_home_2010_11_2017_18.loc[:, column_name_list_oreb_pct_dreb_pct_reb_pct_away_home_game_id_game_date],
             on=column_name_list_merge_on,
             how='inner')

    #fill missing Offensive Rebound Percentage, Defensive Rebound Percentage, and Rebound Percentage away and home of Team Advanced Box Scores
    for column_name_tuple_pair in column_name_tuple_list_OREB_PCT_DREB_PCT_REB_PCT_away_home:
        df_team_advanced_box_scores_away_home_oreb_pct_dreb_pct_reb_pct_away_home.loc[:, column_name_tuple_pair[0]] = \
        df_team_advanced_box_scores_away_home_oreb_pct_dreb_pct_reb_pct_away_home.loc[:, column_name_tuple_pair[0]]\
        .fillna(df_team_advanced_box_scores_away_home_oreb_pct_dreb_pct_reb_pct_away_home.loc[:, column_name_tuple_pair[1]])
        
    
    return df_team_advanced_box_scores_away_home_oreb_pct_dreb_pct_reb_pct_away_home\
           .drop(columns=column_name_list_oreb_pct_dreb_pct_reb_pct_away_home)
   
    
    





def combine_and_clean_team_advanced_box_scores_and_team_box_scores(df_team_advanced_box_scores_away_home,
                                                                   df_team_box_scores_2_1_away_home_2010_11_2017_18):
    
    #combine Team Advanced Box Scores and Team Box Scores
    df_team_advanced_box_scores_away_home_team_box_scores_away_home = \
    pd.merge(df_team_advanced_box_scores_away_home.rename(columns={'TEAM_ID_away':'team_id_away', 'TEAM_ID_home':'team_id_home'}), 
             df_team_box_scores_2_1_away_home_2010_11_2017_18, 
             on=['game_id', 'game_date', 'team_id_away', 'team_id_home'], 
             how='inner') 
    
    return df_team_advanced_box_scores_away_home_team_box_scores_away_home.sort_values(['game_date', 'game_id']).reset_index(drop=True)






##########################################################################################################################
##########################################################################################################################
'''Get Team Advanced Box Scores and Team Box Scores Collection in Stacked, Away Team and Home Team, and Team A and Team B format.'''



def get_stacked_data_frame_from_two_suffix_data_frame(df,
                                                      column_name_suffix_list):
    '''
    Take data frame with two suffixes and return the data frame stacked transformation
    and suffix category column name.'''
    
    #get column name list from data frame and column name suffix list
    column_name_list = list(df.columns)
    
    column_name_list_first_suffix = \
    [k for k in column_name_list if k.endswith(column_name_suffix_list[0])]
    
    column_name_list_second_suffix = \
    [k for k in column_name_list if k.endswith(column_name_suffix_list[1])]
    
    column_name_list_not_first_suffix = \
    [k for k in column_name_list if not k in column_name_list_first_suffix]
    
    column_name_list_not_second_suffix = \
    [k for k in column_name_list if not k in column_name_list_second_suffix]


    column_name_list_first_suffix_stripped = \
    [k.split(column_name_suffix_list[0])[0] for k in column_name_list_first_suffix]
    
    column_name_list_second_suffix_stripped = \
    [k.split(column_name_suffix_list[1])[0] for k in column_name_list_second_suffix]
    
    
    #get column name dictionary
    column_name_dict_first_suffix_stripped = \
    dict(zip(column_name_list_first_suffix, column_name_list_first_suffix_stripped))
    
    column_name_dict_second_suffix_stripped = \
    dict(zip(column_name_list_second_suffix, column_name_list_second_suffix_stripped))

    
    
    df_not_second_suffix = \
    df.loc[:, column_name_list_not_second_suffix].rename(columns=column_name_dict_first_suffix_stripped)
    
    df_not_first_suffix = \
    df.loc[:, column_name_list_not_first_suffix].rename(columns=column_name_dict_second_suffix_stripped)
    
    
    
    #get column name suffix category
    column_name_suffix_category = \
    get_column_name_suffix_category_from_column_name_suffix_list(column_name_suffix_list)

    df_not_second_suffix.loc[:, column_name_suffix_category] = 0
    
    df_not_first_suffix.loc[:, column_name_suffix_category] = 1
    
    
    
    df_stacked = \
    pd.concat([df_not_second_suffix, 
               df_not_first_suffix]).sort_values(['game_date', 'game_id']).reset_index(drop=True)


  
    return df_stacked, column_name_suffix_category



#break up into 'away_home' -> 'stacked' and 'stacked' -> 'a_b'    
def get_a_b_matchup_format_of_team_advanced_box_scores_and_team_box_scores_from_away_home_format_of_team_advanced_box_scores_and_team_box_scores(
    df_team_advanced_box_scores_away_home_team_box_scores_away_home):
    '''
    get _a and _b two suffix format of team advanced box scores and team box scores 
    from _away and _home two suffix format of team advanced box scores and team box scores.'''


    def get_column_name_list_and_list_collection_from_team_advanced_box_scores_away_home_team_box_scores_away_home(
        df_team_advanced_box_scores_away_home_team_box_scores_away_home):
        '''
        get column name list collection from team advanced box scores and team box scores: column_name_list_collection_away_home_empty
        get column name list not with _away or _home suffix: column_name_list_game_level_season_level'''
        
        column_name_list = \
        df_team_advanced_box_scores_away_home_team_box_scores_away_home.columns.to_list()

        column_name_list_collection_away_home_empty = {}

        column_name_list_collection_away_home_empty[0] = \
        [k for k in column_name_list if '_away' in k]

        column_name_list_collection_away_home_empty[1] = \
        [k for k in column_name_list if '_home' in k]

        column_name_list_away_home = \
        column_name_list_collection_away_home_empty[0] + column_name_list_collection_away_home_empty[1]

        column_name_list_game_level_season_level = \
        [k for k in column_name_list if not k in column_name_list_away_home]
        
        column_name_list_collection_away_home_empty[2] = \
        ['game_id'] + column_name_list_collection_away_home_empty[0]

        column_name_list_collection_away_home_empty[3] = \
        ['game_id'] + column_name_list_collection_away_home_empty[1]

        column_name_list_collection_away_home_empty[4] = \
        [k.split('_away')[0] for k in column_name_list_collection_away_home_empty[0]]

        return column_name_list_collection_away_home_empty, column_name_list_game_level_season_level


    def get_column_name_dictionary_collection(column_name_list_collection_away_home_empty):
        '''
        get column name dictionary collection for _away to empty and _home to empty.'''

        column_name_dict_collection_away_home = {}

        column_name_dict_collection_away_home[0] = \
        dict(zip(column_name_list_collection_away_home_empty[0], column_name_list_collection_away_home_empty[4]))

        column_name_dict_collection_away_home[1] = \
        dict(zip(column_name_list_collection_away_home_empty[1], column_name_list_collection_away_home_empty[4]))

        return column_name_dict_collection_away_home



    def get_df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty(
        df_team_advanced_box_scores_away_home_team_box_scores_away_home,
        column_name_list_collection_away_home_empty,
        column_name_dict_collection_away_home):
        '''
        get Team Advanced Box Scores and Team Box Scores away and home collection'''

        #select and store to collection the Team Advanced Box Scores and Team Box Scores away and the Team Advanced Box Scores and Team Box Scores home 
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty = {}
        for i in range(2):
            df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[i] = \
            df_team_advanced_box_scores_away_home_team_box_scores_away_home.loc[:, column_name_list_collection_away_home_empty[i+2]]

            df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[i].loc[:, 'away_home'] = i

            df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[i] = \
            df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[i].rename(columns=column_name_dict_collection_away_home[i])


        #combine the Team Advanced Box Scores and Team Box Scores away and the Team Advanced Box Scores and Team Box Scores home
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2] = \
        pd.concat([df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[0], 
                   df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[1]])


        return df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty



    def get_and_add_game_level_season_level_columns_to_df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty_index_2(
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty,
        df_team_advanced_box_scores_away_home_team_box_scores_away_home,
        column_name_list_game_level_season_level):
        '''
        get and add game level and season level columns to df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].'''

        #get game level and season level columns from Team Advanced Box Scores and Team Box Scores
        df_game_id_game_date_matchup_season_year_season_type_season = \
        df_team_advanced_box_scores_away_home_team_box_scores_away_home.loc[:, column_name_list_game_level_season_level]


        #add game level and season level columns
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2] = \
        pd.merge(df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2],
                 df_game_id_game_date_matchup_season_year_season_type_season,
                 on='game_id',
                 how='inner')

        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2] = \
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].sort_values(['game_date', 'game_id'])

        return df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty



    def extract_and_add_column_name_team_abbreviation_a_and_team_abbreviation_b_as_column_name_a_and_b(
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty):
        '''
        extract and add team abbreviation a and team abbreviation b from matchup column as columns a and b.'''

        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].loc[:, 'a'] = \
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].loc[:, 'matchup'].str.split(' vs. ').str[0].str.strip()

        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].loc[:, 'b'] = \
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].loc[:, 'matchup'].str.split(' vs. ').str[1].str.strip()

        return df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty



    #get column name list collection and list
    column_name_list_collection_away_home_empty, column_name_list_game_level_season_level = \
    get_column_name_list_and_list_collection_from_team_advanced_box_scores_away_home_team_box_scores_away_home(
        df_team_advanced_box_scores_away_home_team_box_scores_away_home)

    #get column name dictionary
    column_name_dict_collection_away_home = \
    get_column_name_dictionary_collection(column_name_list_collection_away_home_empty)

    #get team advanced box scores and team box scores collection of Away Team, Home Team, and Stacked
    df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty = \
    get_df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty(
        df_team_advanced_box_scores_away_home_team_box_scores_away_home=df_team_advanced_box_scores_away_home_team_box_scores_away_home,
        column_name_list_collection_away_home_empty=column_name_list_collection_away_home_empty,
        column_name_dict_collection_away_home=column_name_dict_collection_away_home)
    
    
    
    #get team advanced box scores and team box scores collection Stacked with added columns
    df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty = \
    get_and_add_game_level_season_level_columns_to_df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty_index_2(
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty,
        df_team_advanced_box_scores_away_home_team_box_scores_away_home,
        column_name_list_game_level_season_level)

    #extracted and add column name a and b from matchup column
    df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty = \
    extract_and_add_column_name_team_abbreviation_a_and_team_abbreviation_b_as_column_name_a_and_b(
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty)
    
    
    #build column name list with the game level, the season level, and the team a abbreviation and team b abbreviation
    column_name_list_game_level_season_level_a_b = \
    column_name_list_game_level_season_level + ['a', 'b']

    
    #build column name list collection and column name dictionary collection for team a and team b selection and rename
    column_name_list_team_level = \
    [k for k in df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].columns if not k in column_name_list_game_level_season_level_a_b]


    #get column name list collection at the team level Team A and Team B
    column_name_list_team_level_collection_a_b = \
    {0 : [k + '_a' for k in column_name_list_team_level],
     1 : [k + '_b' for k in column_name_list_team_level]}
    
    column_name_list_team_level_collection_a_b[2] = \
    column_name_list_team_level_collection_a_b[0] + column_name_list_team_level_collection_a_b[1]

    column_name_dict_team_level_collection_a_b = {}

    column_name_dict_team_level_collection_a_b[0] = \
    dict(zip(column_name_list_team_level, column_name_list_team_level_collection_a_b[0]))

    column_name_dict_team_level_collection_a_b[1] = \
    dict(zip(column_name_list_team_level, column_name_list_team_level_collection_a_b[1]))
    


    def get_team_a_rows_and_team_b_rows_of_Team_Advanced_Box_Scores_and_Team_Box_Scores_and_rename_their_column_names(
        df_team_advanced_box_scores_team_box_scores_empty,
        column_name_dict_team_level_collection_a_b):
        '''
        get team a rows and team b rows of Team Advanced Box Scores and Team Box Scores and rename their column names.'''

        df_team_advanced_box_scores_team_box_scores_collection_a_b = {}

        #get team a rows
        df_team_advanced_box_scores_team_box_scores_collection_a_b[0] = \
        df_team_advanced_box_scores_team_box_scores_empty.loc[
            df_team_advanced_box_scores_team_box_scores_empty.loc[:, 'TEAM_ABBREVIATION'] == \
            df_team_advanced_box_scores_team_box_scores_empty.loc[:, 'a'], :]

        #rename columns for team _a box scores
        df_team_advanced_box_scores_team_box_scores_collection_a_b[0] = \
        df_team_advanced_box_scores_team_box_scores_collection_a_b[0]\
        .rename(columns=column_name_dict_team_level_collection_a_b[0])


        #get team b rows
        df_team_advanced_box_scores_team_box_scores_collection_a_b[1] = \
        df_team_advanced_box_scores_team_box_scores_empty.loc[
            df_team_advanced_box_scores_team_box_scores_empty.loc[:, 'TEAM_ABBREVIATION'] == \
            df_team_advanced_box_scores_team_box_scores_empty.loc[:, 'b'], :]

        #rename columns for team _b advanced box scores and team _b box scores
        df_team_advanced_box_scores_team_box_scores_collection_a_b[1] = \
        df_team_advanced_box_scores_team_box_scores_collection_a_b[1]\
        .rename(columns=column_name_dict_team_level_collection_a_b[1])

        return df_team_advanced_box_scores_team_box_scores_collection_a_b

    df_team_advanced_box_scores_team_box_scores_collection_a_b = \
    get_team_a_rows_and_team_b_rows_of_Team_Advanced_Box_Scores_and_Team_Box_Scores_and_rename_their_column_names(
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2],
        column_name_dict_team_level_collection_a_b)

    #combine team a Team Advanced Box Scores and Team Box Scores and 
    #team b Team Advanced Box Scores and Team Box Scores

    df_team_advanced_box_scores_team_box_scores_collection_a_b[2] = \
    pd.merge(df_team_advanced_box_scores_team_box_scores_collection_a_b[0],
             df_team_advanced_box_scores_team_box_scores_collection_a_b[1],
             on=column_name_list_game_level_season_level_a_b,
             how='inner')


    return df_team_advanced_box_scores_team_box_scores_collection_a_b[2]#, df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2]











def get_team_advanced_box_scores_and_team_box_scores_collection_stacked_away_home_a_b(
    df_team_advanced_box_scores_away_home_team_box_scores_away_home):
    '''
    Get Team Advanced Box Scores and Team Box Scores Collection in Stacked, Away Team and Home Team, and Team A and Team B format.'''
    
    #initialize Team Advanced Box Scores and Team Box Scores collection
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b = {}
    
    #get Team Advanced Box Scores and Team Box Scores Stacked
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['stacked'],  column_name_suffix_category = \
    get_stacked_data_frame_from_two_suffix_data_frame(df_team_advanced_box_scores_away_home_team_box_scores_away_home,
                                                          column_name_suffix_list=['_away', '_home'])
    
    #get Team Advanced Box Scores and Team Box Scores Away Team and Home Team
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['away_home'] = \
    df_team_advanced_box_scores_away_home_team_box_scores_away_home
    
    
    #get Team Advanced Box Scores and Team Box Scores Team A and Team B
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['a_b']= \
    get_a_b_matchup_format_of_team_advanced_box_scores_and_team_box_scores_from_away_home_format_of_team_advanced_box_scores_and_team_box_scores(
        df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['away_home'])
    
    
    
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['stacked'],  column_name_suffix_category = \
    get_stacked_data_frame_from_two_suffix_data_frame(df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['a_b'],
                                                          column_name_suffix_list=['_a', '_b'])




    
    return df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b













##########################################################################################################################

def get_two_suffix_column_name_data_frame_from_stacked_data_frame(df_stacked, 
                                                                  column_name_list_not_first_suffix_not_second_suffix, 
                                                                  column_name_indicator, 
                                                                  keep_indicator=False, 
                                                                  column_name_suffix_list=None):
    '''
    Take the stacked data frame and transform its columns with one of two suffixes.'''

    
    if column_name_suffix_list == None:
        column_name_suffix_list = ['_a', '_b']
    
    column_name_list = list(df_stacked.columns)
    
    column_name_list_first_suffix_second_suffix = \
    [k for k in column_name_list if not k in column_name_list_not_first_suffix_not_second_suffix]

    df_first_suffix = \
    df_stacked.loc[(df_stacked.loc[:, column_name_indicator] == 0), :]
    
    df_second_suffix = \
    df_stacked.loc[(df_stacked.loc[:, column_name_indicator] == 1), :]
    
    
    if keep_indicator == False:
        df_first_suffix = \
        df_first_suffix.drop(columns=column_name_indicator)
        
        df_second_suffix = \
        df_second_suffix.drop(columns=column_name_indicator)
    
    
    column_name_list_first_suffix_second_suffix_first = \
    [k + column_name_suffix_list[0] for k in column_name_list_first_suffix_second_suffix]
    
    column_name_list_first_suffix_second_suffix_second = \
    [k + column_name_suffix_list[1] for k in column_name_list_first_suffix_second_suffix]
    
    
    column_name_dict_first_suffix_second_suffix_first = \
    dict(zip(column_name_list_first_suffix_second_suffix, column_name_list_first_suffix_second_suffix_first))
    
    
    column_name_dict_first_suffix_second_suffix_second = \
    dict(zip(column_name_list_first_suffix_second_suffix, column_name_list_first_suffix_second_suffix_second))
    
    df_first_suffix = \
    df_first_suffix.rename(columns=column_name_dict_first_suffix_second_suffix_first)
    
    df_second_suffix = \
    df_second_suffix.rename(columns=column_name_dict_first_suffix_second_suffix_second)
    

    return pd.merge(df_first_suffix, 
                    df_second_suffix, 
                    on=column_name_list_not_first_suffix_not_second_suffix)

######################################################################################################################    
######################################################################################################################







def clean_team_city_proper_metropolitan_area_gdp_and_merge_it_with_geographic_team_city_name(
    df_city_proper_metropolitan_area_gdp,
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name):
    '''
    Get and Clean United States and Canada Team City Proper / Metropolitan Area GDP.'''
    


    #drop United Nations Statistics Division Sub-Region
    df_city_proper_metropolitan_area_gdp = \
    df_city_proper_metropolitan_area_gdp.drop(columns='UNSDsub‑region[4]')

    #filter City Proper Metropolitan Area GDP for Country/Region United States and Canada
    df_city_proper_metropolitan_area_gdp = \
    df_city_proper_metropolitan_area_gdp.loc[(df_city_proper_metropolitan_area_gdp.loc[:, 'Country/Region'] == 'United States') | 
                                             (df_city_proper_metropolitan_area_gdp.loc[:, 'Country/Region'] == 'Canada'), :].reset_index(drop=True)

    
    #get City Proper Metropolitan Area uniques list
    us_ca_city_proper_metro_area_list = \
    df_city_proper_metropolitan_area_gdp.loc[:, 'City proper /Metropolitan area'].drop_duplicates().to_list()


##########################################################################################################################
    #get Geographic Team City Name uniques list
    geographic_team_city_name_list = \
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name\
    .loc[:, 'geographic_team_city_name'].drop_duplicates().to_list()

    #get Geographic Team City Names that are a substring of a United States Canada City Proper Metro Area
    geographic_team_city_name_match_list = \
    [k for k in geographic_team_city_name_list 
     if any(k in string for string in us_ca_city_proper_metro_area_list)]


    geographic_team_city_name_list = \
    [k for k in geographic_team_city_name_list if not k in geographic_team_city_name_match_list]

    def return_string_if_substring_in_list(city_proper_metro_area, us_ca_city_proper_metro_area_list,):
        return_list = \
        [us_ca_city_proper_metro_area_list[index] \
         for index, us_ca_city_proper_metro_area in enumerate(us_ca_city_proper_metro_area_list) 
         if city_proper_metro_area in us_ca_city_proper_metro_area]
        
        if len(return_list) == 1:
            return return_list[0]
        elif len(return_list) > 1:
            return return_list
        elif return_list == 0:
            return 'NaN'
        else:
            return 'Error'


    #get Golden State San Francisco list
    golden_state_san_francisco_list = \
    [geographic_team_city_name_list[1], return_string_if_substring_in_list('San Francisco', 
                                                        us_ca_city_proper_metro_area_list)]

    #get East Rutherford New York list
    east_rutherford_new_york_list = \
    [geographic_team_city_name_list[0], return_string_if_substring_in_list('New York', 
                                                           us_ca_city_proper_metro_area_list)]

    #get Brooklyn New York list
    brooklyn_new_york_list = \
    [geographic_team_city_name_list[2], return_string_if_substring_in_list('New York',
                                                    us_ca_city_proper_metro_area_list)]

    sf_ny_list = [golden_state_san_francisco_list, 
                  east_rutherford_new_york_list, 
                  brooklyn_new_york_list]

    
    df_sf_ny = \
    pd.DataFrame(sf_ny_list, columns=['geographic_team_city_name', 'city_proper_metro_area'])


    
    city_team_city_proper_metro_area_list = \
    [[geographic_team_city_name, return_string_if_substring_in_list(geographic_team_city_name, us_ca_city_proper_metro_area_list)] \
     for geographic_team_city_name in geographic_team_city_name_match_list]

    
    df_geographic_team_city_name_city_proper_metro_area = \
    pd.DataFrame(city_team_city_proper_metro_area_list, 
                 columns=['geographic_team_city_name', 'city_proper_metro_area'])

    df_geographic_team_city_name_city_proper_metro_area = \
    pd.concat([df_sf_ny, 
               df_geographic_team_city_name_city_proper_metro_area], 
               axis=0)

    df_geographic_team_city_name_city_proper_metro_area = \
    df_geographic_team_city_name_city_proper_metro_area.reset_index(drop=True)

    
    #add column city_proper_metro_area
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area = \
    pd.merge(df_game_id_game_date_TEAM_CITY_geographic_team_city_name,
             df_geographic_team_city_name_city_proper_metro_area,
             on=['geographic_team_city_name'],
             how='inner').sort_values(['game_date', 'game_id']).reset_index(drop=True)

    #
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP = \
    pd.merge(df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area,
             df_city_proper_metropolitan_area_gdp.rename(columns= {'City proper /Metropolitan area':'city_proper_metro_area',
                                                                   'GDP ($billions)': 'city_proper_metro_area_GDP_bil', 
                                                                   'GDP ($billions).1': 'city_proper_metro_area_GDP_bil1'}),
             on='city_proper_metro_area').sort_values(['game_date', 'game_id']).reset_index(drop=True)

    #clean up City Proper Metro Area GDP

    #clean up City Proper Metro Area GDP by dropping year and source number
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP.loc[:, 'city_proper_metro_area_GDP_bil'] = \
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP.loc[:, 'city_proper_metro_area_GDP_bil']\
    .str.extract('([^\(:]+)').rename(columns={0:'city_proper_metro_area_GDP_bil'})

    #clean City Proper Metro Area GDP and City Proper Metro Area GDP 1

    #remove commas and spaces from City Proper Metropolitan Area GDP and City Proper Metropolitan Area GDP 1
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP.loc[:, ['city_proper_metro_area_GDP_bil', 'city_proper_metro_area_GDP_bil1']] = \
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP.loc[:, ['city_proper_metro_area_GDP_bil', 'city_proper_metro_area_GDP_bil1']]\
    .replace(',','', regex=True)

    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP.loc[:, ['city_proper_metro_area_GDP_bil', 'city_proper_metro_area_GDP_bil1']] = \
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP.loc[:, ['city_proper_metro_area_GDP_bil', 'city_proper_metro_area_GDP_bil1']].replace(' ','', regex=True)

    #convert type string to type float64 of City Proper Metropolitan Area GDP and City Proper Metropolitan Area GDP 1
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP.loc[:, ['city_proper_metro_area_GDP_bil', 'city_proper_metro_area_GDP_bil1']] = \
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP.loc[:, ['city_proper_metro_area_GDP_bil', 'city_proper_metro_area_GDP_bil1']].astype('float64')

    return df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP
##########################################################################################################################











##########################################################################################################################

'''Feature Engineering functions'''

#################################################################################################################################



def get_first_n_folds_of_data_frame_with_n_splits(df, n_folds, n_splits):
    
    
    from sklearn.model_selection import TimeSeriesSplit
    
    for i, (train_indices, test_indices) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(df)):
        if i == n_folds - 1:
            df_first_n_folds = df.iloc[train_indices]
    
    return df_first_n_folds
    

#################################################################################################################################

def plot_histogram_with_kde_mean_median_mode(df, title):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.histplot(data=df, x='NET_RATING', kde=True, bins=40)

    mia_or_mean=df.loc[:, 'NET_RATING'].mean()
    mia_or_median=df.loc[:, 'NET_RATING'].median()
    mia_or_mode=df.loc[:, 'NET_RATING'].mode().values[0]

    _ = plt.axvline(mia_or_mean, color='r', linestyle='--', label="Mean")
    _ = plt.axvline(mia_or_median, color='g', linestyle='-', label="Median")
    _ = plt.axvline(mia_or_mode, color='b', linestyle='-', label="Mode")
    _ = plt.title(title)


    _ = plt.legend()

    
#################################################################################################################################

def get_team_id_pair_t_statistic_p_value_per_season(df,
                                                    column_name,
                                                    season_list,
                                                    team_id_pairs_list):
    
    from scipy import stats
    df_season_team_id_0_team_id_1_t_statistic_p_value = pd.DataFrame({})
    
    for season in season_list:

        df_season = df.loc[df.loc[:, 'season'] == season]

        for t_ in team_id_pairs_list:
            df_season_team_id_0 = \
            df_season.loc[(df_season.loc[:, 'team_id'] == t_[0]), 
                           :]

            df_season_team_id_1 = \
            df_season.loc[(df_season.loc[:, 'team_id'] == t_[1]), 
                           :]

            t_statistic_p_value_tuple = \
            stats.ttest_ind(df_season_team_id_0.loc[:, column_name],
                            df_season_team_id_1.loc[:, column_name])
            
            t_statistic_p_value_list = list(t_statistic_p_value_tuple)


            df_temp = pd.DataFrame([[season, 
                                      t_[0], 
                                      t_[1], 
                                      t_statistic_p_value_list[0], 
                                      t_statistic_p_value_list[1]]])


            df_season_team_id_0_team_id_1_t_statistic_p_value = \
            pd.concat([df_season_team_id_0_team_id_1_t_statistic_p_value, df_temp])


    df_season_team_id_0_team_id_1_t_statistic_p_value.columns = ['season', 'team_id_0', 'team_id_1', 't_statistic', 'p_value']
    df_season_team_id_0_team_id_1_t_statistic_p_value = df_season_team_id_0_team_id_1_t_statistic_p_value.reset_index(drop=True)
    
    
    return df_season_team_id_0_team_id_1_t_statistic_p_value


#################################################################################################################################

def get_p_value_proportion_by_alpha(df, season_list, team_id_pairs_list, column_name='NET_RATING', alpha=.05):

    df_season_team_id_0_team_id_1_t_statistic_p_value = get_team_id_pair_t_statistic_p_value_per_season(df=df, column_name= column_name, season_list=season_list, team_id_pairs_list=team_id_pairs_list)

    proportion_significant = sum(df_season_team_id_0_team_id_1_t_statistic_p_value.loc[:, 'p_value'] < alpha) / df_season_team_id_0_team_id_1_t_statistic_p_value.shape[0]
    
    proportion_not_significant = sum(df_season_team_id_0_team_id_1_t_statistic_p_value.loc[:, 'p_value'] > alpha) / df_season_team_id_0_team_id_1_t_statistic_p_value.shape[0]
    
    return proportion_significant, proportion_not_significant

#################################################################################################################################

def get_permutation_test_p_value(df,
                                 season_list,
                                 team_id_pairs_list,
                                 column_name='NET_RATING',
                                 number_permutations=10000):
    enumerator = 0
    
    df_season_team_id_0_team_id_1_column_name_p_value = pd.DataFrame({})
    
    for season in season_list:
        df_season_column_name = df.loc[df.loc[:, 'season'] == season, [column_name, 'team_id', 'season']]
        
        for team_id_pair_ in team_id_pairs_list:
            df_season_column_name_team_id_01 = df_season_column_name.loc[(df_season_column_name.loc[:, 'team_id'] == team_id_pair_[0]) | (df_season_column_name.loc[:, 'team_id'] == team_id_pair_[1]), :]

            column_name_permutation = column_name + '_permutation'
            
            operator_difference = np.empty(number_permutations)
            for i in range(number_permutations):
                df_season_column_name_team_id_01.loc[:, column_name_permutation] = np.random.permutation(df_season_column_name_team_id_01.loc[:, column_name])
                
                operator_difference[i] = \
                df_season_column_name_team_id_01.loc[df_season_column_name_team_id_01.loc[:, 'team_id'] == team_id_pair_[0], column_name_permutation].mean() - \
                df_season_column_name_team_id_01.loc[df_season_column_name_team_id_01.loc[:, 'team_id'] == team_id_pair_[1], column_name_permutation].mean()

            observed_operator_difference = \
            df_season_column_name_team_id_01.loc[df_season_column_name_team_id_01.loc[:, 'team_id'] == team_id_pair_[0], column_name].mean() - \
            df_season_column_name_team_id_01.loc[df_season_column_name_team_id_01.loc[:, 'team_id'] == team_id_pair_[1], column_name].mean()
                
                
            observed_operator_difference_absolute_value = abs(observed_operator_difference)
                
            column_name_p_value = np.sum(abs(operator_difference) > observed_operator_difference_absolute_value) / len(operator_difference)

            enumerator += 1
            
            df_temp = pd.DataFrame([[season, team_id_pair_[0], team_id_pair_[1], column_name_p_value]])
            
            df_season_team_id_0_team_id_1_column_name_p_value = pd.concat([df_season_team_id_0_team_id_1_column_name_p_value, df_temp])
            
    df_season_team_id_0_team_id_1_column_name_p_value.columns = ['season', 'team_id_0', 'team_id_1', column_name + '_p_value']
    
    df_season_team_id_0_team_id_1_column_name_p_value = df_season_team_id_0_team_id_1_column_name_p_value.reset_index(drop=True)
    
    return df_season_team_id_0_team_id_1_column_name_p_value


    
#################################################################################################################################


    
    
    

#################################################################################################################################


def get_player_advanced_box_score_lost_contribution(df_player_advanced_box_scores_2009_10_2017_18,
                                                    df_player_injury_report,
                                                    df_player_inactives_2010_11_2017_18):

    def get_player_advanced_box_score_rows_where_player_had_zero_minutes(df_player_advanced_box_scores_2009_10_2017_18):
        '''
        Get player advanced box score rows where player had zero minutes.'''

        return df_player_advanced_box_scores_2009_10_2017_18.loc[(df_player_advanced_box_scores_2009_10_2017_18.loc[:, 'season'] != '2009-10') & 
                                                                 (df_player_advanced_box_scores_2009_10_2017_18.loc[:, 'MIN'] == 0),
                                                                 :]


    df_player_advanced_box_scores_2010_11_2017_18_zero_minutes = \
    get_player_advanced_box_score_rows_where_player_had_zero_minutes(df_player_advanced_box_scores_2009_10_2017_18)




    def get_player_advanced_box_score_rows_where_player_had_nonzero_minutes_and_did_not_get_injured_in_game(df_player_advanced_box_scores_2009_10_2017_18,
                                                                                                            df_player_injury_report):
        '''
        get player advanced box score rows where the player played and was not injured in game..'''

        def get_player_advanced_box_score_rows_where_player_had_nonzero_minutes(df_player_advanced_box_scores_2009_10_2017_18):
            '''
            get player advanced box score rows where player had non zero minutes.'''

            return df_player_advanced_box_scores_2009_10_2017_18.loc[df_player_advanced_box_scores_2009_10_2017_18.loc[:, 'MIN'] != 0, 
                                                                     :]


        def get_player_advanced_box_scores_in_game_injury_implied(df0,
                                                              df1,
                                                              column_name_list_merge_on):

            df_player_advanced_box_scores_player_injury_report_2010_11_2017_18 = \
            pd.merge(df0.rename(columns={'GAME_ID':'game_id', 'TEAM_ID':'team_id'}), 
                     df1, 
                     on=column_name_list_merge_on, 
                     how='inner')

            return df_player_advanced_box_scores_player_injury_report_2010_11_2017_18.loc[df_player_advanced_box_scores_player_injury_report_2010_11_2017_18.loc[:, 'MIN'] != 0, :]\
                                                                        .reset_index(drop=True)


        def get_exclusive_left_join(df0,
                                    df1,
                                    column_name_list_merge_on):
            df = pd.merge(df0, 
                          df1, 
                          on=column_name_list_merge_on, 
                          how='outer', 
                          indicator=True)

            return df.loc[df.loc[:, '_merge'] == 'left_only', :].drop(columns=['_merge'])


        #get player advanced box score rows where player had non zero minutes
        df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes = \
        get_player_advanced_box_score_rows_where_player_had_nonzero_minutes(df_player_advanced_box_scores_2009_10_2017_18)

        #get player advanced box score rows where player was implied injured in game.
        df_player_advanced_box_scores_in_game_injury_implied_2010_11_2017_18 = \
        get_player_advanced_box_scores_in_game_injury_implied(df0=df_player_advanced_box_scores_2009_10_2017_18,
                                                              df1=df_player_injury_report,
                                                              column_name_list_merge_on=['game_id', 'team_id', 'PLAYER_ID', 'season', 'game_date'])

        #get player advanced box score rows where player had non zero minutes and was not implied injured in game.
        df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries = \
        get_exclusive_left_join(df0 = df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes.rename(columns={'GAME_ID':'game_id'}),
                                df1 = df_player_advanced_box_scores_in_game_injury_implied_2010_11_2017_18.loc[:, ['game_id', 'PLAYER_ID']],
                                column_name_list_merge_on = ['game_id', 'PLAYER_ID'])


        return df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries.sort_values(['game_date', 'game_id'])\
                                                                                        .reset_index(drop=True).reset_index()


    df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries = \
    get_player_advanced_box_score_rows_where_player_had_nonzero_minutes_and_did_not_get_injured_in_game(df_player_advanced_box_scores_2009_10_2017_18,
                                                                                                        df_player_injury_report)




    def get_column_name_list_player_advanced_box_score_stat_column_name_list_player_advanced_box_score_stat_CMA_and_column_name_dictionary_player_advanced_box_score_stats_CMA():
        '''
        get player advanced box score stat CMA dictionary and column name lists that make up it.'''

        column_name_list_player_advanced_box_score_stats = \
        ['MIN', 'E_OFF_RATING', 'OFF_RATING',
         'E_DEF_RATING', 'DEF_RATING', 'E_NET_RATING', 'NET_RATING', 'AST_PCT',
         'AST_TOV', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT',
         'EFG_PCT', 'TS_PCT', 'USG_PCT', 'E_USG_PCT', 'E_PACE', 'PACE',
         'PACE_PER40', 'POSS', 'PIE']

        column_name_list_player_advanced_box_score_stats_CMA = \
        [k + '_CMA' for k in column_name_list_player_advanced_box_score_stats]

        column_name_dict_advanced_box_score_stats = dict(zip(column_name_list_player_advanced_box_score_stats, column_name_list_player_advanced_box_score_stats_CMA))
        column_name_dict_advanced_box_score_stats.update({'level_2':'index'})

        return column_name_list_player_advanced_box_score_stats, column_name_list_player_advanced_box_score_stats_CMA, column_name_dict_advanced_box_score_stats

    column_name_list_player_advanced_box_score_stats, column_name_list_player_advanced_box_score_stats_CMA, column_name_dict_advanced_box_score_stats = \
    get_column_name_list_player_advanced_box_score_stat_column_name_list_player_advanced_box_score_stat_CMA_and_column_name_dictionary_player_advanced_box_score_stats_CMA()



    def extract_and_add_player_advanced_box_score_stat_cma(df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries,
                                                           column_name_list_player_advanced_box_score_stats,
                                                           column_name_dict_advanced_box_score_stats):
        '''
        Extract and add player advanced box score stat cma of games player played in and was not injured as player advanced box score stat '_CMA'.'''


        #add groupby column names PLAYER_NAME and season to player advanced box score stats column name list
        column_name_list_player_advanced_box_score_stats_player_name_season = \
        ['PLAYER_NAME', 'season'] + column_name_list_player_advanced_box_score_stats


        #get boxscore advanced stat cma of the ordered dataframe
        df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA = \
        df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries.loc[:, column_name_list_player_advanced_box_score_stats_player_name_season]\
                                                                            .groupby(['PLAYER_NAME', 'season'])\
                                                                            .rolling(99999, min_periods=0).mean().reset_index()\
                                                                            .rename(columns=column_name_dict_advanced_box_score_stats)

        #add player advanced box score stat cma of games player played in and was not injured to data frame
        df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA = \
        pd.merge(df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA, 
                 df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries.drop(columns=['season', 'PLAYER_NAME']), 
                 on='index')

        return df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA


    df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA = \
    extract_and_add_player_advanced_box_score_stat_cma(df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries,
                                                       column_name_list_player_advanced_box_score_stats,
                                                       column_name_dict_advanced_box_score_stats)



    def combine_and_clean_data_frames(df_player_injury_report,
                                      df_player_inactives_2010_11_2017_18,
                                      df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA,
                                      df_player_advanced_box_scores_2010_11_2017_18_zero_minutes):
        '''
        Combine player injury report, 
                player inactive, 
                player advanced box scores with nonzero minutes and not injured in game CMA, and
                player advanced box scores with zero minutes'''


        def pre_concat_rename_data_frame_columns(df_player_injury_report,
                                                 df_player_inactives_2010_11_2017_18,
                                                 df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA,
                                                 df_player_advanced_box_scores_2010_11_2017_18_zero_minutes):
            '''
            Rename data frames before combining.'''

            df_player_injury_report = \
            df_player_injury_report.rename(columns={'PLAYER_NAME':'player', 
                                             'TEAM_ID':'team_id', 
                                             'TEAM_NAME':'Team',
                                             'team':'Team'})

            df_player_inactives_2010_11_2017_18 = \
            df_player_inactives_2010_11_2017_18.rename(columns={'PLAYER_NAME':'player', 
                                                                'TEAM_ID':'team_id', 
                                                                'TEAM_NAME':'Team',
                                                                'team':'Team'})

            df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA = \
            df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA.rename(columns={'PLAYER_NAME':'player', 
                                                                                                                    'TEAM_ID':'team_id', 
                                                                                                                    'TEAM_NAME':'Team',
                                                                                                                    'team':'Team'})
            df_player_advanced_box_scores_2010_11_2017_18_zero_minutes = \
            df_player_advanced_box_scores_2010_11_2017_18_zero_minutes.rename(columns={'PLAYER_NAME':'player', 
                                                                                       'TEAM_ID':'team_id', 
                                                                                       'TEAM_NAME':'Team',
                                                                                       'GAME_ID':'game_id',
                                                                                       'team':'Team'})

            return df_player_injury_report, \
                   df_player_inactives_2010_11_2017_18, \
                   df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA, \
                   df_player_advanced_box_scores_2010_11_2017_18_zero_minutes

        #rename data frame columns before data frame concat
        df_player_injury_report,\
        df_player_inactives_2010_11_2017_18,\
        df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA,\
        df_player_advanced_box_scores_2010_11_2017_18_zero_minutes = \
        pre_concat_rename_data_frame_columns(df_player_injury_report,
                                             df_player_inactives_2010_11_2017_18,
                                             df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA,
                                             df_player_advanced_box_scores_2010_11_2017_18_zero_minutes)

        #combine data frames
        df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives = \
        pd.concat([df_player_injury_report, 
                   df_player_inactives_2010_11_2017_18, 
                   df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA, 
                   df_player_advanced_box_scores_2010_11_2017_18_zero_minutes]).sort_values(['game_date', 'game_id']).reset_index(drop=True)




        def drop_player_advanced_box_scores_player_injury_report_duplicates_of_player_inactives(
            df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives):
            '''
            Clean combined data frame of player advanced box score, player injury report, and player inactives.'''

            def get_indices_of_advanced_box_scores_player_injury_report_duplicating_player_inactives_rows(df):

                df_duplicated = df.loc[df.loc[:, ['game_id', 'team_id', 'PLAYER_ID']].duplicated(keep=False), :]

                low_information_duplicate_indices_list = list(df_duplicated.loc[df_duplicated.loc[:, 'TEAM_CITY'].isnull(), :].index)

                return low_information_duplicate_indices_list    


            advanced_box_scores_player_injury_report_drop_indices_list = \
            get_indices_of_advanced_box_scores_player_injury_report_duplicating_player_inactives_rows(df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives)

            df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives = \
            df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives.drop(advanced_box_scores_player_injury_report_drop_indices_list)

            return df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives.sort_values(['game_date', 'game_id']).reset_index(drop=True)


        df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives = \
        drop_player_advanced_box_scores_player_injury_report_duplicates_of_player_inactives(df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives)

        return df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives


    df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives = \
    combine_and_clean_data_frames(df_player_injury_report,
                                  df_player_inactives_2010_11_2017_18,
                                  df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries_CMA_and_before_CMA,
                                  df_player_advanced_box_scores_2010_11_2017_18_zero_minutes)




    def get_column_name_list_player_advanced_box_score_stats_CMA_game_id_game_date_team_id_season_PLAYER_ID(column_name_list_player_advanced_box_score_stats_CMA):

        column_name_list_player_advanced_box_score_stats_CMA_game_id_game_date_team_id_season_PLAYER_ID = \
        column_name_list_player_advanced_box_score_stats_CMA.copy()

        column_name_list_player_advanced_box_score_stats_CMA_game_id_game_date_team_id_season_PLAYER_ID.extend(['game_id', 'game_date', 'team_id', 'season', 'PLAYER_ID'])

        return column_name_list_player_advanced_box_score_stats_CMA_game_id_game_date_team_id_season_PLAYER_ID


    column_name_list_player_advanced_box_score_stats_CMA_game_id_game_date_team_id_season_PLAYER_ID = \
    get_column_name_list_player_advanced_box_score_stats_CMA_game_id_game_date_team_id_season_PLAYER_ID(column_name_list_player_advanced_box_score_stats_CMA)






    def fill_forward_player_advanced_box_score_nonzero_minutes_no_injuries_CMA_to_advanced_box_scores_zero_minutes_CMA_player_injury_report_CMA_player_inactives_CMA(
        df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives,
        column_name_list_player_advanced_box_score_stats_CMA_game_id_game_date_team_id_season_PLAYER_ID):
        '''
        fill forward player advanced box score nonzero minutes and no injury CMA to 
        player advanced box scores zero minutes, player injury report, and player inactives with NaN advanced box score stat CMA.'''


        df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives.loc[:, column_name_list_player_advanced_box_score_stats_CMA_game_id_game_date_team_id_season_PLAYER_ID] = \
        df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives.groupby('PLAYER_ID')\
                                                                                                                                                   [column_name_list_player_advanced_box_score_stats_CMA_game_id_game_date_team_id_season_PLAYER_ID]\
                                                                                                                                                   .transform(lambda v: v.ffill())

        df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_forward_filled_player_advanced_box_scores_zero_minutes_CMA_player_injury_report_CMA_player_inactives_CMA = \
        df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives.copy()

        return df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_forward_filled_player_advanced_box_scores_zero_minutes_CMA_player_injury_report_CMA_player_inactives_CMA


    df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_forward_filled_player_advanced_box_scores_zero_minutes_CMA_player_injury_report_CMA_player_inactives_CMA = \
    fill_forward_player_advanced_box_score_nonzero_minutes_no_injuries_CMA_to_advanced_box_scores_zero_minutes_CMA_player_injury_report_CMA_player_inactives_CMA(
        df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives,
        column_name_list_player_advanced_box_score_stats_CMA_game_id_game_date_team_id_season_PLAYER_ID)





    def get_and_clean_rows_of_forward_filled_player_advanced_box_score_stat_zero_minutes_player_injury_report_and_player_inactives(
        df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries,
        df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_forward_filled_player_advanced_box_scores_zero_minutes_CMA_player_injury_report_CMA_player_inactives_CMA,
        column_name_list_player_advanced_box_score_stats_CMA):

        def get_rows_of_forward_filled_player_advanced_box_score_stat_zero_minutes_player_injury_report_and_player_inactives(
                df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries,
                df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_forward_filled_player_advanced_box_scores_zero_minutes_CMA_player_injury_report_CMA_player_inactives_CMA):
            '''
            filter for player lost contribution rows of player advanced box scores, player injury report, and player inactives.'''

            def right_exclusive_join(df0,
                                     df1,
                                     column_name_list_merge_on):

                df_merged = pd.merge(df0,
                                     df1,
                                     on=column_name_list_merge_on, 
                                     how='outer', 
                                     indicator=True)

                return df_merged.loc[df_merged.loc[:, '_merge'] == 'right_only', :].reset_index(drop=True)


            df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives = \
            right_exclusive_join(df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries.loc[:, ['game_id', 
                                                                                                              'TEAM_ID', 
                                                                                                              'PLAYER_ID']].rename(columns={'TEAM_ID':'team_id'}),
                                 df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives,
                                 column_name_list_merge_on=['game_id', 'team_id', 'PLAYER_ID'])

            return df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives

        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives = \
        get_rows_of_forward_filled_player_advanced_box_score_stat_zero_minutes_player_injury_report_and_player_inactives(
                df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries,
                df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_forward_filled_player_advanced_box_scores_zero_minutes_CMA_player_injury_report_CMA_player_inactives_CMA)



        def clean_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_data_frame(df,
                                                                                                                         columns_checked):
            '''
            remove game-player rows where a player advanced box score stat CMA are null after the fill forward on and deemed useless.'''

            return df.loc[~df.loc[:, columns_checked].isnull().all(axis=1), :].reset_index(drop=True)


        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives = \
        clean_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_data_frame(
            df=df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives,
            columns_checked=column_name_list_player_advanced_box_score_stats_CMA)


        return df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives


    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives = \
    get_and_clean_rows_of_forward_filled_player_advanced_box_score_stat_zero_minutes_player_injury_report_and_player_inactives(
        df_player_advanced_box_scores_2009_10_2017_18_nonzero_minutes_no_injuries,
        df_player_advanced_box_scores_nonzero_minutes_no_injuries_CMA_forward_filled_player_advanced_box_scores_zero_minutes_CMA_player_injury_report_CMA_player_inactives_CMA,
        column_name_list_player_advanced_box_score_stats_CMA)
    
    return column_name_list_player_advanced_box_score_stats,\
           column_name_list_player_advanced_box_score_stats_CMA,\
           df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives,
           



    
    
    
    
 
    

def get_team_advanced_box_score_lost_contribution_sum_mean_and_max(df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives,
                                                                   column_name_list_player_advanced_box_score_stats,
                                                                   column_name_list_player_advanced_box_score_stats_CMA):    
    
    
    def get_lost_contribution_sum_mean_or_max(df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives,
                                          column_name_list_player_advanced_box_score_stats,
                                          column_name_list_player_advanced_box_score_stats_CMA,
                                          operator='sum'):


        column_name_list_team_advanced_box_score_stats_lost_contribution_operator = \
        ['lc_' + str(operator) + '_' + k for k in column_name_list_player_advanced_box_score_stats]

        column_name_dict_advanced_box_score_stats_lost_contribution_operator = \
        dict(zip(column_name_list_player_advanced_box_score_stats_CMA, 
                 column_name_list_team_advanced_box_score_stats_lost_contribution_operator))


        #get team lost contribution sum, mean, or max per forward filled player advanced box score zero minutes, player injury report, and player inactive
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_operator = \
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives.groupby(['game_id', 'team_id', 'game_date', 'season'])\
                                                                                                      [column_name_list_player_advanced_box_score_stats_CMA].agg(operator)\
                                                                                                      .rename(columns=column_name_dict_advanced_box_score_stats_lost_contribution_operator).reset_index()

        return df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_operator


    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum = \
    get_lost_contribution_sum_mean_or_max(df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives,
                                          column_name_list_player_advanced_box_score_stats,
                                          column_name_list_player_advanced_box_score_stats_CMA,
                                          operator='sum')
    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_mean = \
    get_lost_contribution_sum_mean_or_max(df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives,
                                          column_name_list_player_advanced_box_score_stats,
                                          column_name_list_player_advanced_box_score_stats_CMA,
                                          operator='mean')

    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_max = \
    get_lost_contribution_sum_mean_or_max(df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives,
                                          column_name_list_player_advanced_box_score_stats,
                                          column_name_list_player_advanced_box_score_stats_CMA,
                                          operator='max')

    def get_df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max(
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum,
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_mean,
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_max):
        '''
        merge df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum, _mean, and _max.'''


        data_frames = [df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum,
                       df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_mean,
                       df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_max]
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max = \
        reduce(lambda  left, right: pd.merge(left,
                                             right,
                                             on=['game_id', 'team_id', 'game_date', 'season'],
                                             how='inner'),
               data_frames)
        return df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max

    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max = \
    get_df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max(
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum,
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_mean,
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_max)
    
    return df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max






def clean_df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max(
    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max, 
    df_player_active_2009_10_2017_18_player_inactives_2009_10_2017_18):
    

    def add_team_game_rows_without_a_lost_contribution_and_fillna_zero(
        df_game_id_game_date_team_team_id_season_2010_11_2017_18,
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max):


        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max = \
        pd.merge(df_game_id_game_date_team_team_id_season_2010_11_2017_18, 
                 df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max, 
                 on=['game_id', 'game_date', 'team_id', 'season'],
                 how='left')

        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max = \
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max.fillna(0)

        return df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max

    
    season_list = \
    ['2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18']

    df_game_id_game_date_team_team_id_season_2010_11_2017_18 = \
    filter_data_frame_by_season(df_player_active_2009_10_2017_18_player_inactives_2009_10_2017_18,
                                season_list).loc[:, ['game_id', 'game_date', 'team', 'team_id', 'season']].drop_duplicates()\
                                            .sort_values(['game_date', 'game_id']).reset_index(drop=True)

    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max = \
    add_team_game_rows_without_a_lost_contribution_and_fillna_zero(
        df_game_id_game_date_team_team_id_season_2010_11_2017_18,
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max)

    return df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max










def get_and_clean_team_advanced_box_score_stat_lost_contribution_sum_mean_max(df_player_advanced_box_scores_2009_10_2017_18,
                                                                              df_player_injury_report,
                                                                              df_player_inactives_2010_11_2017_18,
                                                                             df_player_active_2009_10_2017_18_player_inactives_2009_10_2017_18):
    
    
    column_name_list_player_advanced_box_score_stats, \
    column_name_list_player_advanced_box_score_stats_CMA, \
    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives = \
    get_player_advanced_box_score_lost_contribution(df_player_advanced_box_scores_2009_10_2017_18,
                                                    df_player_injury_report,
                                                    df_player_inactives_2010_11_2017_18)

    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max = \
    get_team_advanced_box_score_lost_contribution_sum_mean_and_max(df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives,
                                                                       column_name_list_player_advanced_box_score_stats,
                                                                       column_name_list_player_advanced_box_score_stats_CMA)

    
    
    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max = \
    clean_df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max(
        df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max,
        df_player_active_2009_10_2017_18_player_inactives_2009_10_2017_18)
    
    return df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max
##########################################################################################################################









    
####################################################################################################################    
#Get Numeric Operator Group Name Operator Window Max Difference
    
def get_column_name_suffix_category_from_column_name_suffix_list(column_name_suffix_list):
    '''
    get suffix category column name: suffix_category.'''

    alphanumeric = ''
    column_name_suffix_category = ''
    for suffix in column_name_suffix_list:
        for character in suffix:
            if character.isalnum():
                alphanumeric += character

        column_name_suffix_category = column_name_suffix_category + alphanumeric + '_'
        alphanumeric = ''

    return column_name_suffix_category



def get_numeric_group_operator_window_max_data_frame_from_stacked_data_frame(df_stacked,
                                                                             column_name_list_not_difference,
                                                                             column_name_list_groupby,
                                                                             window_size_list_max_min,
                                                                             column_name_suffix_category,
                                                                             operator_group_name,
                                                                             operator_name):

    #get column name list for column group cma and column group shift
    column_name_list_numeric = list(df_stacked.select_dtypes('number'))

    column_name_list_numeric_not_not_difference_not_suffix_category = \
    [k for k in column_name_list_numeric if not k in column_name_list_not_difference and 
                                            not k in column_name_suffix_category]

    #get column name list for all column names in groupby window moving average.
    column_name_list_numeric_not_not_difference_not_suffix_category_groupby = \
    [k for k in column_name_list_numeric_not_not_difference_not_suffix_category if not k in column_name_list_groupby] + column_name_list_groupby

    
    
    
    #get column name list for Window Moving Average of Data Frame
    
    if operator_name == 'mean':
        cumulative_operator_name = '_cma'
    elif operator_name == 'max':
        cumulative_operator_name = '_cmax'
    elif operator_name == 'sum':
        cumulative_operator_name = '_csum'
    elif operator_name == 'min':
        cumulative_operator_name = '_cmin'
    elif operator_name == 'median':
        cumulative_operator_name = '_cmedian'
    elif operator_name == 'std':
        cumulative_operator_name = '_cstd'
    elif operator_name == 'var':
        cumulative_operator_name = '_cvar'
#     elif operator_name == 'mode':
#         cumulative_operator_name = '_cmode'

        
    
    if operator_group_name != None and operator_group_name != '':
        column_name_list_numeric_not_not_difference_not_suffix_category_cma_window_max = \
        [k + '_' + operator_group_name + cumulative_operator_name + str(window_size_list_max_min[0]) for k in column_name_list_numeric_not_not_difference_not_suffix_category]
    else:
        column_name_list_numeric_not_not_difference_not_suffix_category_cma_window_max = \
        [k + cumulative_operator_name + str(window_size_list_max_min[0]) for k in column_name_list_numeric_not_not_difference_not_suffix_category]
    
    
    
    
    #get column name dictionary for rename of Window Moving Average of Data Frame
    column_name_dict_numeric_not_not_difference_not_suffix_category_cma_window_max = \
    dict(zip(column_name_list_numeric_not_not_difference_not_suffix_category, 
             column_name_list_numeric_not_not_difference_not_suffix_category_cma_window_max))

    
    #get column group window moving average and column group shift
    df_stacked_numerics_not_not_difference_not_suffix_category_group_cma_window_max_groupby_index = \
    df_stacked.loc[:, column_name_list_numeric_not_not_difference_not_suffix_category_groupby]\
    .groupby(column_name_list_groupby).rolling(window_size_list_max_min[0],
                                   window_size_list_max_min[1]).agg(operator_name)\
    .groupby(column_name_list_groupby).shift(periods=1)


    mulitindex_number_of_levels = \
    df_stacked_numerics_not_not_difference_not_suffix_category_group_cma_window_max_groupby_index.index.nlevels


    #rename columns
    df_stacked_numerics_not_not_difference_not_suffix_category_group_cma_window_max_groupby_index = \
    df_stacked_numerics_not_not_difference_not_suffix_category_group_cma_window_max_groupby_index\
    .rename(columns=column_name_dict_numeric_not_not_difference_not_suffix_category_cma_window_max).reset_index()


    df_stacked_numerics_not_not_difference_not_suffix_category_group_cma_window_max_groupby_index = \
    df_stacked_numerics_not_not_difference_not_suffix_category_group_cma_window_max_groupby_index\
    .rename(columns={'level_' + str(mulitindex_number_of_levels - 1): 'index'})

    return df_stacked_numerics_not_not_difference_not_suffix_category_group_cma_window_max_groupby_index, column_name_list_numeric_not_not_difference_not_suffix_category






def get_data_frame_numeric_difference(df_two_suffix, 
                                      column_name_list_groupby, 
                                      column_name_list_not_difference, 
                                      column_name_suffix_list):
    '''
    #get numeric cma window max column difference by first suffix and second suffix.'''
    
    
    #get column name list
    column_name_list_numeric = \
    list(df_two_suffix.select_dtypes('number').columns)
    
    column_name_list_numeric = \
    [k for k in column_name_list_numeric if not k in column_name_list_not_difference]
    
    #get column name list numeric cma window max with suffix stripped
    column_name_list_numeric_cma_window_max_stripped_of_suffix = \
    [k.split(column_name_suffix_list[0])[0] for k in column_name_list_numeric if k.endswith(column_name_suffix_list[0])]
    
    column_name_list_numeric_cma_window_max_stripped_of_suffix_not_groupby = \
    [k for k in column_name_list_numeric_cma_window_max_stripped_of_suffix if not k in column_name_list_groupby]

    #get column name list numeric cma window max with first suffix
    column_name_list_numeric_cma_window_max_first_suffix_not_groupby = \
    [k + column_name_suffix_list[0] for k in column_name_list_numeric_cma_window_max_stripped_of_suffix_not_groupby]
    
    #get column name list numeric cma window max with second suffix
    column_name_list_numeric_cma_window_max_second_suffix_not_groupby = \
    [k + column_name_suffix_list[1] for k in column_name_list_numeric_cma_window_max_stripped_of_suffix_not_groupby]
    
    #get column name list numeric cma window max with difference suffix
    column_name_list_numeric_cma_window_max_difference_not_groupby = \
    [k + '_diff' for k in column_name_list_numeric_cma_window_max_stripped_of_suffix_not_groupby]
    
    #get column name list numeric cma window max with one of two suffixes
    column_name_list_numeric_cma_window_max_first_suffix_second_suffix_not_groupby = \
    column_name_list_numeric_cma_window_max_first_suffix_not_groupby + column_name_list_numeric_cma_window_max_second_suffix_not_groupby
    
    
    df_two_suffix.loc[:, column_name_list_numeric_cma_window_max_difference_not_groupby] = \
    df_two_suffix.loc[:, column_name_list_numeric_cma_window_max_first_suffix_not_groupby].fillna(0).values - \
    df_two_suffix.loc[:, column_name_list_numeric_cma_window_max_second_suffix_not_groupby].fillna(0).values
    
    
    #drop column name numeric moving average with one of two suffixes
    return df_two_suffix.drop(columns=column_name_list_numeric_cma_window_max_first_suffix_second_suffix_not_groupby)




def from_team_a_team_b_or_stacked_format_of_team_advanced_box_scores_and_team_box_scores_get_operator_window_max_difference(
    df,
    column_name_list_groupby,
    column_name_list_not_difference,
    column_name_suffix_list,
    window_size_list_max_min,
    operator_group_name,
    operator_name,
    data_frame_start_format):
    '''
    From team a and team b format of data frame Team Advanced Box Scores and Team Box Scores get Window Max Game Moving Average Difference data frame.'''
    

    if data_frame_start_format == 'a_b':
        #Get stacked data frame from two suffix data frame.
        df_stacked, column_name_suffix_category = \
        get_stacked_data_frame_from_two_suffix_data_frame(df, 
                                                          column_name_suffix_list)
    elif data_frame_start_format == 'stacked':
        column_name_suffix_category = 'a_b_'
        df_stacked = df
    

    
    #Get Numerics Game Max Moving Average, Groupby, and Index data frame from stacked data frame
    df_stacked_numerics_cma_window_max_groupby_index, column_name_list_numeric_not_not_difference_not_suffix_category = \
    get_numeric_group_operator_window_max_data_frame_from_stacked_data_frame(df_stacked,
                                                                             column_name_list_not_difference,
                                                                             column_name_list_groupby,
                                                                             window_size_list_max_min,
                                                                             column_name_suffix_category,
                                                                             operator_group_name,
                                                                             operator_name)
    
    
    #Get Not Numerics, Groupby, and Index data frame from stacked data frame
    column_name_list_numeric_not_not_difference_not_suffix_category_not_groupby = \
    [k for k in column_name_list_numeric_not_not_difference_not_suffix_category if not k in column_name_list_groupby]
    
    df_stacked_index = df_stacked.reset_index()
    
    df_stacked_not_numerics_groupby_index = \
    df_stacked_index.drop(columns=column_name_list_numeric_not_not_difference_not_suffix_category_not_groupby)
    
    column_name_list_groupby_index = ['index'] + column_name_list_groupby
    
    

    #combine Not Numerics and Numerics Game Max Moving Average by merging on Groupby and Index
    df_stacked_numerics_cma_window_max_not_numerics_groupby = \
    pd.merge(df_stacked_not_numerics_groupby_index, 
             df_stacked_numerics_cma_window_max_groupby_index, 
             on=column_name_list_groupby_index,
             how='inner').drop(columns='index')
     

    #get two suffix data frame from stacked data frame
    df_two_suffix = get_two_suffix_column_name_data_frame_from_stacked_data_frame(df_stacked=df_stacked_numerics_cma_window_max_not_numerics_groupby, column_name_list_not_first_suffix_not_second_suffix=column_name_list_not_difference, column_name_indicator=column_name_suffix_category, keep_indicator=False, column_name_suffix_list=None).sort_values(['game_date', 'game_id']).reset_index(drop=True)


    #take difference of Numeric columns with the same prefix, store to numeric_column_name_operator_window_max_diff
    return get_data_frame_numeric_difference(df_two_suffix, 
                                             column_name_list_groupby, 
                                             column_name_list_not_difference, 
                                             column_name_suffix_list)

##########################################################################################################################










##########################################################################################################################
def get_operator_window_max_5_12_999_difference_of_team_player_advanced_box_score_lost_contribution_sum_mean_max(
    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max,
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b,
    operator_name):
    '''
    get Operator Window Max 5, 12, and 999 Difference of Team Player Advanced Box Score Lost Contribution Sum, Mean, Max.'''

    #add column names 'matchup' and 'a_b_' to Team Player Advanced Box Score Lost Contribution Sum, Mean, and Max
    column_name_list_game_level_season_level = \
    [k for k in df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['stacked'].columns 
     if k in ['game_id', 'game_date','team_id', 'TEAM_NAME','matchup', 'season', 'a_b_']]

    column_name_list_merge_on = \
    [k for k in column_name_list_game_level_season_level 
     if k in df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max] + ['team']


    df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max = \
    pd.merge(df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['stacked']\
             .loc[:, column_name_list_game_level_season_level].rename(columns={'TEAM_NAME':'team'}),
             df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max,
             on=column_name_list_merge_on,
             how='inner')


    #get Operator Window Max of Team Player Advanced Box Score Lost Contribution Sum, Mean, and Max 
    df_team_player_advanced_box_score_lost_contribution_sum_mean_max_operarator_window_max_5_12_999_diff_collection = {}

    for window_max in [5, 12, 999]:
        df_team_player_advanced_box_score_lost_contribution_sum_mean_max_operarator_window_max_5_12_999_diff_collection[str(window_max)] = \
        df_team_advanced_box_score_season_matchup_cma12_diff_team_box_score_season_matchup_cma12_diff_a_b = \
        from_team_a_team_b_or_stacked_format_of_team_advanced_box_scores_and_team_box_scores_get_operator_window_max_difference(
            df=df_forward_filled_player_advanced_box_scores_zero_minutes_player_injury_report_player_inactives_lost_contribution_sum_mean_max,
            column_name_list_groupby=['season', 'team_id'],
            column_name_list_not_difference=['game_id', 'game_date', 'matchup','season'],
            column_name_suffix_list=['_a', '_b'],
            window_size_list_max_min=[window_max,1],
            operator_group_name='season',
            operator_name=operator_name,
            data_frame_start_format='stacked')


    #get column name list to merge on
    column_name_list_merge_on = \
    [k for k in df_team_player_advanced_box_score_lost_contribution_sum_mean_max_operarator_window_max_5_12_999_diff_collection['5'] 
     if k in df_team_player_advanced_box_score_lost_contribution_sum_mean_max_operarator_window_max_5_12_999_diff_collection['12']]


    #merge Team Player Advanced Box Score Lost Contribution Sum, Mean, and Max Operator Window Max 5, 12, and 999 Differences

    #get data frame list of Team Player Advanced Box Score Lost Contribution Sum, Mean, Max Operator Window Max 5, 12, 999 Difference
    data_frame_list = \
    [df_team_player_advanced_box_score_lost_contribution_sum_mean_max_operarator_window_max_5_12_999_diff_collection[key] for key in ['5', '12', '999']]

    #merge Team Player Lost Contribution Sum Mean Max Operator Window Max Difference data frame list
    df_team_player_advanced_box_score_lost_contribution_sum_mean_max_cma5_12_999_diff = \
    reduce(lambda  left, right : pd.merge(left,
                                          right,
                                          on=column_name_list_merge_on,
                                          how='inner'), 
           data_frame_list)
    
    return df_team_player_advanced_box_score_lost_contribution_sum_mean_max_cma5_12_999_diff
#################################################################################################################################






#################################################################################################################################
def plot_kde(column_name_x, column_name_y='score_difference_a', df_game_id_column_y=None, df_game_id_folds_012345678=None, df_game_id_column_x=None):
    
    import matplotlib.pyplot as plt
    savefig_relative_file_path = 'figures/figure_' + str(column_name_y) + '_vs_' + str(column_name_x) + '_folds_012345678.png'
    
    readback = return_figure_if_it_exists(filename=savefig_relative_file_path)

    if isinstance(readback, np.ndarray) == True:
        #show the image
        import matplotlib.image as mpimg
        img = mpimg.imread(savefig_relative_file_path)
        plt.figure(figsize=(8, 6), dpi=100)
        plt.imshow(img)
        plt.axis('off')

    else:
        #filter for column_name_y folds 0 thru 8 
        df_game_id_column_name_y_folds_012345678 = pd.merge(df_game_id_column_y.loc[:, ['game_id', column_name_y]], df_game_id_folds_012345678, on='game_id', how='inner')

        #add column_name_x
        df_game_id_column_name_y_column_name_x_folds_012345678 = pd.merge(df_game_id_column_name_y_folds_012345678, df_game_id_column_x.loc[:, ['game_id', column_name_x]], on='game_id', how='inner')

        #plot kde
        import seaborn as sns
        import matplotlib.pyplot as plt
        pp = sns.jointplot(x=df_game_id_column_name_y_column_name_x_folds_012345678.loc[:, column_name_x], 
                           y=df_game_id_column_name_y_column_name_x_folds_012345678.loc[:, column_name_y], 
                           kind='kde', 
                           shade=True,
                           color='orange',
                           cmap="Oranges",
                           thresh=False)

        pp.fig.suptitle(str(column_name_y) + ' vs. ' + str(column_name_x))

        pp.ax_joint.collections[0].set_alpha(0)
        pp.fig.tight_layout()
        pp.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

        plt.savefig(fname=savefig_relative_file_path, bbox_inches="tight")
#################################################################################################################################
















#################################################################################################################################
def get_team_advanced_box_score_team_box_score_season_matchup_cmax12_difference_season_cma12_season_matchup_cmax12_difference_for_team_a_and_team_b(
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b):
    '''
    Get Season Matchup 12 Game Max Difference of Team Advanced Box Scores, Team Box Scores, Team Advanced Box Scores 
    Season 12 Game Moving Average, and Team Box Scores 12 Game Moving Average.'''
    

    #get Team Advanced Box Score and Team Box Score Season 12 Game Moving Average
    df_team_advanced_box_score_season_cma12_team_box_score_season_cma12_index, column_name_list_numeric_not_not_difference_not_suffix_category = \
    get_numeric_group_operator_window_max_data_frame_from_stacked_data_frame(
        df_stacked=df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['stacked'],
        column_name_list_not_difference=['game_id', 'game_date', 'season', 'matchup','season_type', 'season_year', 'a', 'b'],
        column_name_list_groupby=['season', 'team_id'],
        window_size_list_max_min=[12,1],
        column_name_suffix_category=['a_b_'],
        operator_group_name='season',
        operator_name='mean')

    #add Team Advanced Box Score and Team Box Score Season 12 Game Moving Average to 
    #    Team Advanced Box Score and Team Box Score Season
    df_team_advanced_box_score_season_empty_cma12_team_box_score_season_empty_cma12 = \
    pd.merge(
        df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['stacked'].reset_index(),
        df_team_advanced_box_score_season_cma12_team_box_score_season_cma12_index,
        on=['season', 'team_id','index'],
        how='inner').drop(columns=['index'])


    #get Season Matchup 12 Game Max Difference of Team Advanced Box Score, Team Box Score,  
    #                                             Team Advanced Box Score Season 12 Game Moving Average, and Team Box Score Season 12 Game Moving Average
    df_team_advanced_box_score_team_box_score_season_matchup_cmax12_diff_and_season_cma12_season_matchup_cmax12_diff_a_b = \
    from_team_a_team_b_or_stacked_format_of_team_advanced_box_scores_and_team_box_scores_get_operator_window_max_difference(
        df=df_team_advanced_box_score_season_empty_cma12_team_box_score_season_empty_cma12,
        column_name_list_groupby=['season', 'matchup', 'team_id'],
        column_name_list_not_difference=['game_id', 'game_date', 'season', 'season_type', 'season_year', 'a', 'b', 'matchup'],
        column_name_suffix_list=['_a', '_b'],
        window_size_list_max_min=[12,1],
        operator_group_name='season_matchup',
        operator_name='max',
        data_frame_start_format='stacked')

    return df_team_advanced_box_score_team_box_score_season_matchup_cmax12_diff_and_season_cma12_season_matchup_cmax12_diff_a_b
##########################################################################################################################









##########################################################################################################################
def get_strength_of_schedule_and_related_features(df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b):
    '''
    get data frame for calculation of Strength of Schedule from Team Advanced Box Scores and Team Box Scores collection index stacked.'''

    
    #get column name list for selection from Team Advanced Box Scores and Team Box Scores collection index stacked
    column_name_list = \
    ['game_id', 'game_date', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'team_id', 'matchup', 'season', 'wl']

    #get Strength of Schedule Collection
    df_team_strength_of_schedule_collection_stacked_a_b_difference = {}

    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'] = \
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['stacked'].loc[:, column_name_list]



    #extract and add Team A and Team B columns a and b
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'a'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'matchup'].str.split(' vs. ').str[0].str.strip()

    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'b'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'matchup'].str.split(' vs. ').str[1].str.strip()

    #extract and add suffix indicator column _a and _b
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked']\
    .loc[df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'a'] == \
         df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'TEAM_ABBREVIATION'], '_a'] = 1

    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked']\
    .loc[df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'a'] != \
         df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'TEAM_ABBREVIATION'], '_a'] = 0

    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked']\
    .loc[df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'b'] == \
         df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'TEAM_ABBREVIATION'], '_b'] = 1

    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked']\
    .loc[df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'b'] != \
         df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, 'TEAM_ABBREVIATION'], '_b'] = 0

    #extract and add indicator variables wl_L or wl_W
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'] = \
    pd.concat([df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'], \
               pd.get_dummies(df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, ['wl']])], 
              axis=1)
    #rename column name to loss or win
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].rename(columns={'wl_L':'loss', 'wl_W':'win'})



    #extract columns wins, losses, and win percentage from team_id, season, loss, and win
    df_game_id_team_id_season_loss_win = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, ['game_id', 'team_id', 'season', 'loss', 'win']]

    df_game_id_team_id_season_losses_wins = \
    df_game_id_team_id_season_loss_win.groupby(['team_id', 'season']).rolling(999, 1).agg('sum')\
                                      .groupby(['team_id', 'season']).shift(periods=1, fill_value=0)[['loss', 'win']]

    df_game_id_team_id_season_losses_wins = \
    df_game_id_team_id_season_losses_wins.reset_index()

    df_game_id_team_id_season_losses_wins = \
    df_game_id_team_id_season_losses_wins.rename(columns={'level_2': 'index', 
                                                          'loss':'losses', 
                                                          'win':'wins'})

    df_game_id_team_id_season_losses_wins.loc[:, 'win_percentage'] = \
    df_game_id_team_id_season_losses_wins.loc[:, 'wins'] / \
    (df_game_id_team_id_season_losses_wins.loc[:, 'wins'] + df_game_id_team_id_season_losses_wins.loc[:, 'losses'])



    #add back Team wins, losses, and win percentage.
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].reset_index()

    #get column name list of columns in common
    column_name_list_merge_on = \
    [k for k in df_game_id_team_id_season_losses_wins.columns 
     if k in df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].columns]

    #merge back calculated team wins, losses, and win_percentage
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'] = \
    pd.merge(df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'], 
             df_game_id_team_id_season_losses_wins, 
             on=column_name_list_merge_on)

    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].drop(columns=['index', '_a'])

    #pl(df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].columns.to_list())

    #get column name list at the game level and season level
    column_name_list_not_first_suffix_not_second_suffix = \
    [k for k in df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].columns 
     if k in ['game_id', 'game_date', 'matchup', 'season_year', 'season_type', 'season', 'a', 'b']]


    #convert Strength of Schedule Related Features Stack to Team A and Team B
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'] = \
    get_two_suffix_column_name_data_frame_from_stacked_data_frame(
        df_stacked=df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'], 
        column_name_list_not_first_suffix_not_second_suffix=column_name_list_not_first_suffix_not_second_suffix, 
        column_name_indicator='_b')


    #extract and add Team A opponent losses, opponent wins, and opponent win percentage 
    #and Team B opponent losses, opponent wins, and opponent win percentage
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponent_win_percentage_a'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'win_percentage_b']

    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponent_win_percentage_b'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'win_percentage_a']

    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponent_losses_a'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'losses_b']

    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponent_wins_a'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'wins_b']

    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponent_losses_b'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'losses_a']

    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponent_wins_b'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'wins_a']


    #convert Strength of Schedule Related Features to Stacked format from Team A and Team B format
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'], column_name_suffix_category = \
    get_stacked_data_frame_from_two_suffix_data_frame(df=df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'],
                                                          column_name_suffix_list=['_a', '_b'])

    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].reset_index()


    #calculate Team Opponent season cumalative losses count.
    df_team_id_season_opponents_losses_opponents_wins = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].loc[:, ['team_id', 'season', 'opponent_losses', 'opponent_wins']]\
    .groupby(['team_id', 'season']).rolling(999, 1).agg('sum')\
    .reset_index().rename(columns={'level_2':'index',
                                   'opponent_losses':'opponents_losses',
                                   'opponent_wins':'opponents_wins'})


    #add back Opponents Losses and Opponents Wins to Strength of Schedule Collection
    df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'] = \
    pd.merge(df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'],
             df_team_id_season_opponents_losses_opponents_wins,
             on=['team_id', 'season', 'index'])



    #get column name list at the game level and season level.
    column_name_list_not_first_suffix_not_second_suffix = \
    [k for k in df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].columns 
     if k in ['game_id', 'game_date', 'matchup', 'season_year', 'season_type', 'season', 'a', 'b']]


    #convert Strength of Schedule data frame from Team A and Team B format to Stacked.
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'] = \
    get_two_suffix_column_name_data_frame_from_stacked_data_frame(df_team_strength_of_schedule_collection_stacked_a_b_difference['stacked'].drop(columns=['index']), 
                                                                      column_name_list_not_first_suffix_not_second_suffix=column_name_list_not_first_suffix_not_second_suffix, 
                                                                      column_name_indicator='a_b_')

    #rename columns 
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b']\
    .rename(columns={'opponents_losses_a':'opponents_opponents_losses_b',
                     'opponents_wins_a':'opponents_opponents_wins_b',
                     'opponents_losses_b':'opponents_opponents_losses_a',
                     'opponents_wins_b':'opponents_opponents_wins_a'})

    #calculate and store Team Opponents' Opponents Win Percentage for Team A and Team B
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponents_opponents_win_percentage_a'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponents_opponents_wins_a'] / \
    (df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponents_opponents_losses_a'] + 
     df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponents_opponents_wins_a'])

    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponents_opponents_win_percentage_b'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponents_opponents_wins_b'] / \
    (df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponents_opponents_losses_b'] + \
     df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponents_opponents_wins_b'])


    #calculate and store Team Strength of Schedule for Team a and Team b
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'strength_of_schedule_a'] = \
    (2 * df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponent_win_percentage_a'] +\
     df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponents_opponents_win_percentage_a']) / 3

    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'strength_of_schedule_b'] = \
    (2 * df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponent_win_percentage_b'] + \
     df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].loc[:, 'opponents_opponents_win_percentage_b']) / 3


    #fill Team Strength of Schedule and Related Features missing values with 0 before taking team difference
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'] = \
    df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'].fillna(0)

    #get column name list to not take the difference for
    column_name_list_game_level_season_level = \
    [k for k in df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'] if not k.endswith('_a') and
                                                                                         not k.endswith('_b')]

    #get Team Strength of Schedule and Related Features Difference
    df_team_strength_of_schedule_collection_stacked_a_b_difference['difference'] = \
    get_data_frame_numeric_difference(df_team_strength_of_schedule_collection_stacked_a_b_difference['a_b'],
                                      column_name_list_groupby=['team_id', 'season'],
                                      column_name_list_not_difference=column_name_list_game_level_season_level,
                                      column_name_suffix_list=['_a', '_b'])
    
    return df_team_strength_of_schedule_collection_stacked_a_b_difference
##########################################################################################################################













##########################################################################################################################
def get_geographic_team_city_name_latitude_and_longitude_of_geographic_team_city_name(
    df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff):
    '''
    get Geographic Team City Name Latitude and Longitude of Geographic Team City Name.'''

    #get Geographic Team City Name list
    geographic_team_city_name_list = \
    df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff\
    .loc[:, 'geographic_team_city_name'].drop_duplicates().to_list()


    #get Geographic Team City Name Latitude list and Longitude list
    # from Geographic Team City Name
    geographic_team_city_name_latitude_list = []
    geographic_team_city_name_longitude_list = []

    for address in geographic_team_city_name_list:
        geolocator = Nominatim(user_agent="Your_Name")
        location = geolocator.geocode(address)
        #print(location.address)
        #print((location.latitude, location.longitude))
        geographic_team_city_name_latitude_list += [location.latitude]
        geographic_team_city_name_longitude_list += [location.longitude]



    #convert Geographic Team City Name list, 
    #        Geographic Team City Name Latitude list,
    #        Geographic Team City Name Longitude list
    #to a data frame

    df_geographic_team_city_name_latitude_longitude = \
    pd.DataFrame(zip(geographic_team_city_name_list, 
                     geographic_team_city_name_latitude_list, 
                     geographic_team_city_name_longitude_list),
                 columns = ['geographic_team_city_name', 
                            'geographic_team_city_name_latitude', 
                            'geographic_team_city_name_longitude'])

    return df_geographic_team_city_name_latitude_longitude





def get_team_city_to_game_city_travel_distance_and_time_zone_difference(
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP,
    df_geographic_team_city_name_latitude_longitude):
    '''
    get travel distance from team city to game city and time zone difference.'''
    
    
    #add Geographic Team City Name Latitude and Longitude to df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff
    df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff = \
    pd.merge(df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP,
             df_geographic_team_city_name_latitude_longitude,
             on='geographic_team_city_name',
             how='inner').sort_values(['game_date', 'game_id']).reset_index(drop=True)

    #get Game Geographic City Name Latitude and Longitude
    df_game_id_game_geographic_city_name_latitude_longitude = \
    df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff\
    .loc[df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff.loc[:, 'away_home'] == 1, 
         ['game_id', 
          'geographic_team_city_name_latitude', 
          'geographic_team_city_name_longitude']]\
    .rename(columns = {'geographic_team_city_name_latitude':'geographic_game_city_name_latitude',
                       'geographic_team_city_name_longitude':'geographic_game_city_name_longitude'})


    #add geographic game city name latitude and longitude
    df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff = \
    pd.merge(df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff,
             df_game_id_game_geographic_city_name_latitude_longitude,
             on='game_id',
             how='inner')


    #get team city coordinate list
    team_city_coordinates_list = \
    list(zip(
        df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff.loc[:, 'geographic_team_city_name_latitude'], 
        df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff.loc[:, 'geographic_team_city_name_longitude']))

    #get game city coordinate list
    game_city_coordinates_list = list(zip(
        df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff.loc[:, 'geographic_game_city_name_latitude'], 
        df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff.loc[:, 'geographic_game_city_name_longitude']))

    #get travel distance list
    travel_distance_list = \
    [hs.haversine(loc1,loc2,unit=Unit.MILES) for loc1, loc2 in zip(team_city_coordinates_list, game_city_coordinates_list)]

    #build travel distance data frame
    df_travel_distance = \
    pd.DataFrame({'travel_distance': travel_distance_list})

    #add column travel distance to df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance
    df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff = \
    pd.concat([df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff,
               df_travel_distance],
              axis=1)


    #create time zone finder object
    time_zone_finder_object = TimezoneFinder()

    #get team city and game city time zone list
    team_city_time_zone_list = \
    [time_zone_finder_object.timezone_at(lng=k[1], lat=k[0]) for k in team_city_coordinates_list]

    game_city_time_zone_list = \
    [time_zone_finder_object.timezone_at(lng=k[1], lat=k[0]) for k in game_city_coordinates_list]

    #build team city and game city time zone data frame
    df_team_city_time_zone_game_city_time_zone = \
    pd.DataFrame({'team_city_time_zone':team_city_time_zone_list,
                  'game_city_time_zone':game_city_time_zone_list})


    #Add Team City Time Zone and Game City Time Zone to df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff
    df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff = \
    pd.concat([df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff,
               df_team_city_time_zone_game_city_time_zone,],
             axis=1)


    def time_zone_difference(date, time_zone1, time_zone2):
        '''
        Returns the difference in hours between timezone1 and timezone2
        for a given date.
        '''
        date = pd.to_datetime(date)
        return (time_zone1.localize(date) - \
                time_zone2.localize(date).astimezone(time_zone1)).seconds/3600 + \
                (time_zone1.localize(date) - \
                time_zone2.localize(date).astimezone(time_zone1)).days * 24


    #get Team City-Game City Time Zone Difference list
    time_zone_difference_list = \
    [time_zone_difference('06-06-06', timezone(team_city_time_zone), timezone(game_city_time_zone)) 
     for team_city_time_zone, game_city_time_zone in zip(team_city_time_zone_list, game_city_time_zone_list)]


    #build time zone difference data frame
    df_time_zone_difference = \
    pd.DataFrame({'time_zone_diff':time_zone_difference_list})

    #add time zone difference to df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff
    df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff = \
    pd.concat([df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff,
               df_time_zone_difference], 
              axis=1)

    #extract and add Team City-Game City Time Zone Difference Absolute Value from Team City-Game City Time Zone Difference
    df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff.loc[:, 'time_zone_diff_abs'] = \
    abs(df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff.loc[:, 'time_zone_diff'])

    return df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff




def get_geographic_team_city_name_latitude_longitude_travel_distance_diff_and_time_zone_diff(
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP):
    

    df_geographic_team_city_name_latitude_longitude = \
    get_geographic_team_city_name_latitude_and_longitude_of_geographic_team_city_name(
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP)


    df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff = \
    get_team_city_to_game_city_travel_distance_and_time_zone_difference(
        df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP,
        df_geographic_team_city_name_latitude_longitude)
    
    return df_game_id_game_date_geographic_team_city_name_empty_latitude_longitude_travel_distance_time_zone_diff
##########################################################################################################################











##########################################################################################################################
def get_team_back_to_back_game_season_sum_window_max_2_5_12_999_diff_back_to_back_game_season_mean_window_max_2_5_12_999_diff(
    df_stacked):
    
    
    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff = \
    df_stacked.loc[:, ['game_id', 'game_date', 'team_id', 'TEAM_ABBREVIATION', 'TEAM_NAME','season', 'a_b_']]
    
    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff.loc[:, 'game_date_diff'] = \
    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff.loc[:, ['team_id', 'season', 'game_date']]\
    .groupby(['team_id', 'season']).diff(periods=1).rename({'game_date':'game_date_diff'})
    
    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff.loc[:, 'game_date_diff_1_day'] = 0
    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff\
    .loc[df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff\
         .loc[:, 'game_date_diff'] == pd.Timedelta(seconds=60*60*24), 'game_date_diff_1_day'] = 1
    
    
    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff_a_b = {}
    
    df_game_id_game_date_team_id_season_back_to_back_game_mean_window_max_2_5_12_999_diff_a_b = {}
    
    for window_max in [2, 5, 12, 999]:
        df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff_a_b[str(window_max)] = \
        from_team_a_team_b_or_stacked_format_of_team_advanced_box_scores_and_team_box_scores_get_operator_window_max_difference(
            df=df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff,
            column_name_list_groupby=['team_id', 'season'],
            column_name_list_not_difference=['game_id', 'game_date', 'season'],
            column_name_suffix_list=['_a', '_b'],
            window_size_list_max_min=[window_max, 1],
            operator_group_name='season',
            operator_name='sum',
            data_frame_start_format='stacked')
        
        
        
        df_game_id_game_date_team_id_season_back_to_back_game_mean_window_max_2_5_12_999_diff_a_b[str(window_max)] = \
        from_team_a_team_b_or_stacked_format_of_team_advanced_box_scores_and_team_box_scores_get_operator_window_max_difference(
            df=df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff,
            column_name_list_groupby=['team_id', 'season'],
            column_name_list_not_difference=['game_id', 'game_date', 'season'],
            column_name_suffix_list=['_a', '_b'],
            window_size_list_max_min=[window_max, 1],
            operator_group_name='season',
            operator_name='mean',
            data_frame_start_format='stacked')
        
        
        
    #fix game_date_diff    
    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff.loc[:, 'game_date_diff'] = \
    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff.loc[:, 'game_date_diff'].dt.days

    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff.loc[:, 'game_date_diff'] = \
    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff.loc[:, 'game_date_diff'].fillna(9.0)

    
    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff.loc[:, 'game_date_diff'] = \
    df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff.loc[:, 'game_date_diff'].astype('int64')

    #return df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff
    
    #get df_game_date_diff_diff_game_date_diff1_diff

    #get two suffix format
    df_game_id_game_date_team_id_season_game_date_diff_game_date_diff1_a_b = \
    get_two_suffix_column_name_data_frame_from_stacked_data_frame(
        df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff,
        column_name_list_not_first_suffix_not_second_suffix=['game_id', 'game_date', 'season'], 
        column_name_indicator='a_b_', 
        keep_indicator=False, 
        column_name_suffix_list=None)

    #return df_game_id_game_date_team_id_season_game_date_diff_game_date_diff1_a_b

    #get difference
    df_game_id_game_date_team_id_season_game_date_diff_diff_game_date_diff1_diff = \
    get_data_frame_numeric_difference_basic(
        df_two_suffix=df_game_id_game_date_team_id_season_game_date_diff_game_date_diff1_a_b,
        column_name_list_not_difference=['game_id', 'game_date', 'team_id_a', 'team_id_b'], 
        column_name_suffix_list=['_a', '_b'])
            
        
    data_frame_list = [df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff_a_b[key] \
                       for key in df_game_id_game_date_team_id_season_back_to_back_game_count_window_max_2_5_12_999_diff_a_b.keys()] + \
                      [df_game_id_game_date_team_id_season_back_to_back_game_mean_window_max_2_5_12_999_diff_a_b[key] \
                       for key in df_game_id_game_date_team_id_season_back_to_back_game_mean_window_max_2_5_12_999_diff_a_b.keys()] + \
                      [df_game_id_game_date_team_id_season_game_date_diff_diff_game_date_diff1_diff]

    df_season_back_to_back_game_count_window_max_2_5_12_999_diff_season_back_to_back_game_mean_window_max_2_5_12_999_diff_a_b = \
    reduce(lambda  left, right, : pd.merge(left,
                                           right,
                                           on=[column_name for column_name in left if column_name in right],
                                           how='inner'), 
    data_frame_list)
    
    return df_season_back_to_back_game_count_window_max_2_5_12_999_diff_season_back_to_back_game_mean_window_max_2_5_12_999_diff_a_b
##########################################################################################################################






##########################################################################################################################
def get_away_home_game_count_percentage_day_window(df, 
                                                   day_window):

    
    df.loc[:, 'home'] = df.loc[:, 'away_home']
    df.loc[:, 'away'] = abs(df.loc[:, 'away_home'] - 1)
    df = df.set_index('game_date')
    
    
    #get away, home, and empty game count window max column name collection
    away_home_empty_game_count_day_window_max_column_name_collection = {}
    away_home_empty_game_count_day_window_max_column_name_dict_collection = {}
    away_home_game_percentage_day_window_column_name_collection = {}
        
    df_collection_away_home_empty_game_count_day_window_max_away_home_game_count_percentage_day_window_max = {}
    
    prefix_list = ['away', 'home', 'empty']
    
    
    for prefix in prefix_list:
        #build away home and empty Game Count Day Window column name collection
        if (prefix == 'away') | (prefix == 'home'):
            away_home_empty_game_count_day_window_max_column_name_collection[prefix] = \
            prefix + '_game_day_window_csum' + day_window[0:len(day_window) - 1]
            
            away_home_game_percentage_day_window_column_name_collection[prefix] = \
            prefix + '_game_percent_day_window' + day_window[0:len(day_window) - 1]
            
            #build dictionary collection for away and home column name prefix
            away_home_empty_game_count_day_window_max_column_name_dict_collection[prefix] = {prefix : away_home_empty_game_count_day_window_max_column_name_collection[prefix]}
            
            
            df_collection_away_home_empty_game_count_day_window_max_away_home_game_count_percentage_day_window_max[prefix] = \
            df.loc[:, ['team_id', 'season', prefix]].groupby(['team_id', 'season']).rolling(day_window).agg('sum').reset_index()\
                                                    .rename(columns=away_home_empty_game_count_day_window_max_column_name_dict_collection[prefix])
            
        elif prefix == 'empty':
            away_home_empty_game_count_day_window_max_column_name_collection['empty'] = \
            'game_day_window_csum' + day_window[0:len(day_window) - 1]
            
            #build dictionary collection for empty column name prefix
            away_home_empty_game_count_day_window_max_column_name_dict_collection[prefix] = \
            {'away_home' : away_home_empty_game_count_day_window_max_column_name_collection[prefix]}
            
            
            df_collection_away_home_empty_game_count_day_window_max_away_home_game_count_percentage_day_window_max[prefix] = \
            df.loc[:, ['team_id', 'season', 'away_home']].groupby(['team_id', 'season']).rolling(day_window).agg('count').reset_index()\
                                                        .rename(columns=away_home_empty_game_count_day_window_max_column_name_dict_collection[prefix])
    

    
    #get list of data frame to merge on
    data_frame_list = [df_collection_away_home_empty_game_count_day_window_max_away_home_game_count_percentage_day_window_max[key] 
                       for key in df_collection_away_home_empty_game_count_day_window_max_away_home_game_count_percentage_day_window_max.keys()]

    #merge on the common column name for each left and right data frame
    df_team_id_season_away_home_empty_game_count_day_window = \
    merge_data_frame_list(data_frame_list)
    

    key_list = ['away', 'home']
    
    for key in key_list:
        #calculate percentage of away games in day window. #calculate percentage of home games in day window
        df_team_id_season_away_home_empty_game_count_day_window.loc[:, away_home_game_percentage_day_window_column_name_collection[key]] = \
        df_team_id_season_away_home_empty_game_count_day_window.loc[:, away_home_empty_game_count_day_window_max_column_name_collection[key]] /\
        df_team_id_season_away_home_empty_game_count_day_window.loc[:, away_home_empty_game_count_day_window_max_column_name_collection['empty']].values

        
    #return df_team_id_season_away_home_empty_game_count_day_window
    #merge back to original data frame
    df_team_id_season_away_home_empty_game_count_day_window = pd.merge(df, 
                                                                       df_team_id_season_away_home_empty_game_count_day_window, 
                                                                       on=['team_id', 'season', 'game_date'])
    
    return df_team_id_season_away_home_empty_game_count_day_window
########################################################################################################################








########################################################################################################################


def get_future_game_count_percentage_away_home_both_day_window_max(df_stacked, 
                                                                   day_window_max):
    
    df_stacked.loc[:, 'home'] = df_stacked.loc[:, 'away_home']
    df_stacked.loc[:, 'away'] = abs(df_stacked.loc[:, 'away_home'] - 1)
    df_stacked = df_stacked.set_index('game_date')
    

    column_name_collection_future_away_home_empty_game_count_day_window_max = {}
    column_name_dict_collection_future_away_home_empty_game_count_day_window_max = {}
    df_collection_future_away_home_empty_game_count_day_window_max_future_away_home_game_count_percentage_day_window_max = {}
    
    column_name_collection_future_away_home_game_percentage_day_window_max = {}
    
    
    prefix_list = ['away', 'home', 'empty']
    data_frame_list = []
    
    for prefix in prefix_list:
        if (prefix == 'away') | (prefix == 'home'):
            
            column_name_collection_future_away_home_empty_game_count_day_window_max[prefix] = \
            'future_' + prefix + '_game_count_day_window' + day_window_max[0:len(day_window_max) - 1]
            
            #build column name dictionary
            column_name_dict_collection_future_away_home_empty_game_count_day_window_max[prefix] = \
            {prefix:column_name_collection_future_away_home_empty_game_count_day_window_max[prefix]}
            
            #get future away and home game count per window max
            df_collection_future_away_home_empty_game_count_day_window_max_future_away_home_game_count_percentage_day_window_max[prefix] = \
            df_stacked.loc[:, ['team_id','season', prefix]].groupby(['team_id', 'season'])\
                      .rolling(day_window_max).agg('sum').reset_index()\
                      .rename(columns=column_name_dict_collection_future_away_home_empty_game_count_day_window_max[prefix])
            
            
            data_frame_list += [df_collection_future_away_home_empty_game_count_day_window_max_future_away_home_game_count_percentage_day_window_max[prefix]]
            
        elif prefix == 'empty':

            column_name_collection_future_away_home_empty_game_count_day_window_max[prefix] = \
            'future_game_count_day_window' + day_window_max[0:len(day_window_max) - 1]
            
            
            #build column name dictionary for future away home or empty game count per window max 
            column_name_dict_collection_future_away_home_empty_game_count_day_window_max[prefix] = \
            {'away_home':column_name_collection_future_away_home_empty_game_count_day_window_max[prefix]}
            
            #get future game count per window max
            df_collection_future_away_home_empty_game_count_day_window_max_future_away_home_game_count_percentage_day_window_max[prefix] = \
            df_stacked.loc[:, ['team_id','season', 'away_home']].groupby(['team_id', 'season'])\
                      .rolling(day_window_max).agg('count').reset_index()\
                      .rename(columns=column_name_dict_collection_future_away_home_empty_game_count_day_window_max[prefix])
            
            data_frame_list += [df_collection_future_away_home_empty_game_count_day_window_max_future_away_home_game_count_percentage_day_window_max[prefix]]
        

    #merge data frames in data frame list
    df_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max = \
    reduce(lambda  left, right, : pd.merge(left,
                                           right,
                                           on=[column_name for column_name in left if column_name in right],
                                           how='inner'), 
           data_frame_list)
    
    
    prefix_list = ['away', 'home']
    for prefix in prefix_list:
        
        column_name_collection_future_away_home_game_percentage_day_window_max[prefix] = \
        'future_' + prefix + '_game_percent_day_window' + day_window_max[0:len(day_window_max) - 1]

        df_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max\
        .loc[:, column_name_collection_future_away_home_game_percentage_day_window_max[prefix]] = \
        df_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max\
        .loc[:, column_name_collection_future_away_home_empty_game_count_day_window_max[prefix]] /\
        df_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max\
        .loc[:, column_name_collection_future_away_home_empty_game_count_day_window_max['empty']].values


    df_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max = \
    pd.merge(df_stacked,
             df_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max,
             on=['team_id', 'season', 'game_date'],
             how='inner')
    
    
    return df_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max

########################################################################################################################





##########################################################################################################################

def get_data_frame_numeric_difference_basic(df_two_suffix,
                                            column_name_list_not_difference, 
                                            column_name_suffix_list):
    '''
    #get numeric column difference by first suffix and second suffix.'''
    
    
    #get column name list
    column_name_list_numeric = \
    df_two_suffix.select_dtypes('number').columns.to_list()
    
    column_name_list_numeric_not_not_difference = \
    [k for k in column_name_list_numeric if not k in column_name_list_not_difference]
  
    #get column name list numeric with suffix stripped
    column_name_list_numeric_not_not_difference_stripped_of_suffix = \
    [k.removesuffix(column_name_suffix_list[0]) for k in column_name_list_numeric_not_not_difference if k.endswith(column_name_suffix_list[0])]

    #get column name list numeric with first suffix
    column_name_list_numeric_not_not_difference_first_suffix = \
    [k + column_name_suffix_list[0] for k in column_name_list_numeric_not_not_difference_stripped_of_suffix]
    
    #get column name list numeric with second suffix
    column_name_list_numeric_not_not_difference_second_suffix = \
    [k + column_name_suffix_list[1] for k in column_name_list_numeric_not_not_difference_stripped_of_suffix]
    
    #get column name list numeric with difference suffix
    column_name_list_numeric_not_not_difference_diff = \
    [k + '_diff' for k in column_name_list_numeric_not_not_difference_stripped_of_suffix]
    
    #get column name list numeric with one of two suffixes
    column_name_list_numeric_not_not_difference_first_suffix_second_suffix = \
    column_name_list_numeric_not_not_difference_first_suffix + column_name_list_numeric_not_not_difference_second_suffix

    ###
    df_two_suffix.loc[:, column_name_list_numeric_not_not_difference_diff] = \
    df_two_suffix.loc[:, column_name_list_numeric_not_not_difference_first_suffix].fillna(0).values - \
    df_two_suffix.loc[:, column_name_list_numeric_not_not_difference_second_suffix].fillna(0).values
    
    
    #drop column name numeric moving average with one of two suffixes
    return df_two_suffix.drop(columns=column_name_list_numeric_not_not_difference_first_suffix_second_suffix)



##########################################################################################################################



##########################################################################################################################
    
    
def get_city_proper_metro_area_GDP_diff_geographic_team_city_name_latitude_diff_longitude_diff_travel_distance_diff_time_zone_diff_diff(
    df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP):
    '''get GDP_diff, geographic_team_city_name_latitude_diff _longitude_diff, travel_distance_diff, time_zone_diff_diff'''

    df_game_id_game_date_city_proper_metro_area_GDP_geographic_team_city_name_latitude_longitude_travel_distance_time_zone_diff = \
    get_geographic_team_city_name_latitude_longitude_travel_distance_diff_and_time_zone_diff(
        df_game_id_game_date_TEAM_CITY_geographic_team_city_name_city_proper_metro_area_GDP)

    #save it
    df_game_id_game_date_city_proper_metro_area_GDP_geographic_team_city_name_latitude_longitude_travel_distance_time_zone_diff = \
    save_and_return_data_frame(df=df_game_id_game_date_city_proper_metro_area_GDP_geographic_team_city_name_latitude_longitude_travel_distance_time_zone_diff, 
                               filename='63city_proper_metro_area_GDP_geographic_team_city_name_latitude_longitude_travel_distance_time_zone_diff_2010_2018.csv.gz',
                               index=False,
                               parse_dates=['game_date'])

    #convert to two suffix
    df_game_id_game_date_city_proper_metro_area_GDP_geographic_team_city_name_latitude_longitude_travel_distance_time_zone_diff_a_b = \
    get_two_suffix_column_name_data_frame_from_stacked_data_frame(
        df_stacked=df_game_id_game_date_city_proper_metro_area_GDP_geographic_team_city_name_latitude_longitude_travel_distance_time_zone_diff,
        column_name_list_not_first_suffix_not_second_suffix=['game_id','game_date', 'geographic_game_city_name_latitude', 'geographic_game_city_name_longitude'],
        column_name_indicator='a_b_',
        column_name_suffix_list=['_a', '_b'])

    #get difference
    df_city_proper_metro_area_GDP_diff_geographic_team_city_name_latitude_diff_longitude_diff_travel_distance_diff_time_zone_diff_diff = \
    get_data_frame_numeric_difference_basic(
        df_two_suffix=df_game_id_game_date_city_proper_metro_area_GDP_geographic_team_city_name_latitude_longitude_travel_distance_time_zone_diff_a_b,
        column_name_list_not_difference=['game_id', 'away_home', 'team_id_a', 'team_id_b'],
        column_name_suffix_list=['_a', '_b'])

    return df_city_proper_metro_area_GDP_diff_geographic_team_city_name_latitude_diff_longitude_diff_travel_distance_diff_time_zone_diff_diff
    

##########################################################################################################################


##########################################################################################################################
#get_away_home_empty_game_count_day_window_2_5_8_12_999_diff_away_home_game_count_percentage_day_window_2_5_8_12_999_diff

def get_away_home_empty_game_count_day_window_2_5_8_12_999_diff_away_home_game_count_percentage_day_window_2_5_8_12_999_diff(
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b):
    data_frame_list = []
    day_window_list = [2, 5, 8, 12, 999]

    for day_window in day_window_list:
        data_frame_list += \
        [get_away_home_game_count_percentage_day_window(df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['stacked']\
                                                             .loc[:, ['game_id', 'game_date','team_id', 'season', 'away_home', 'a_b_']], 
                                                             str(day_window) + 'D')]

    df_away_home_empty_game_count_day_window_2_5_8_12_999_away_home_game_count_percentage_day_window_2_5_8_12_999 = \
    merge_data_frame_list(data_frame_list)


    #save it
    df_away_home_empty_game_count_day_window_2_5_8_12_999_away_home_game_count_percentage_day_window_2_5_8_12_999 = \
    save_and_return_data_frame(df=df_away_home_empty_game_count_day_window_2_5_8_12_999_away_home_game_count_percentage_day_window_2_5_8_12_999, 
                               filename='66away_home_empty_game_count_day_window_2_5_8_12_999_away_home_game_count_percentage_day_window_2_5_8_12_999.csv.gz',
                               index=False,
                               parse_dates=['game_date'])


    #get two suffix format
    df_away_home_empty_game_count_day_window_2_5_8_12_999_away_home_game_count_percentage_day_window_2_5_8_12_999_a_b = \
    get_two_suffix_column_name_data_frame_from_stacked_data_frame(
        df_stacked=df_away_home_empty_game_count_day_window_2_5_8_12_999_away_home_game_count_percentage_day_window_2_5_8_12_999,
        column_name_list_not_first_suffix_not_second_suffix=['game_id','game_date', 'season'],
        column_name_indicator='a_b_',
        column_name_suffix_list=['_a', '_b'])


    #get numeric column name differences
    df_away_home_empty_game_count_day_window_2_5_8_12_999_diff_away_home_game_count_percentage_day_window_2_5_8_12_999_diff = \
    get_data_frame_numeric_difference_basic(df_two_suffix=df_away_home_empty_game_count_day_window_2_5_8_12_999_away_home_game_count_percentage_day_window_2_5_8_12_999_a_b,
                                       column_name_list_not_difference=['game_id', 'team_id_a', 'team_id_b', 'game_date', 'season'], 
                                       column_name_suffix_list=['_a', '_b'])

    return df_away_home_empty_game_count_day_window_2_5_8_12_999_diff_away_home_game_count_percentage_day_window_2_5_8_12_999_diff

##########################################################################################################################



    
    

##########################################################################################################################


def get_future_away_home_empty_game_count_day_window_2_5_8_12_999_diff_future_away_home_game_percentage_day_window_2_5_8_12_999_diff(
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b):

    df_playoffs_regular_season = \
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['stacked']\
    .loc[:, ['game_id', 'game_date', 'team_id', 'season', 'away_home', 'season_type', 'a_b_']]


    df_playoffs_regular_season.loc[:, 'home'] = df_playoffs_regular_season.loc[:, 'away_home']
    df_playoffs_regular_season.loc[:, 'away'] = abs(df_playoffs_regular_season.loc[:, 'away_home'] - 1)


    df_playoffs = \
    df_playoffs_regular_season.query("season_type == 'Playoffs'")


    df_regular_season = \
    df_playoffs_regular_season.query("season_type == 'Regular Season'")\
    .sort_values(['game_date', 'game_id'], ascending=False)


    day_window_list = [2, 5, 8, 12, 999]

    df_regular_season_collection = {}
    data_frame_list = []

    for day_window in day_window_list:
        df_regular_season_collection[str(day_window)] = \
        get_future_game_count_percentage_away_home_both_day_window_max(df_regular_season, 
                                                                           str(day_window) + 'D')

        data_frame_list += [df_regular_season_collection[str(day_window)]]


    #combine each future away home empty game count day window max and future away home game percentage window max data frame
    df_regular_season_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max = \
    merge_data_frame_list(data_frame_list)

    df_regular_season_playoffs_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max = \
    pd.concat([df_playoffs,
               df_regular_season_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max], axis=0).fillna(0)


    df_regular_season_playoffs_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max = \
    df_regular_season_playoffs_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max\
    .sort_values(['game_date', 'game_id']).reset_index(drop=True)



    #save it
    df_regular_season_playoffs_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max = \
    save_and_return_data_frame(df=df_regular_season_playoffs_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max, 
                               filename='66future_away_home_empty_game_count_day_window_2_5_8_12_999_future_away_home_game_percentage_day_window_2_5_8_12_999.csv.gz',
                               index=False,
                               parse_dates=['game_date'])





    #convert to two suffix
    df_regular_season_playoffs_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max_a_b = \
    get_two_suffix_column_name_data_frame_from_stacked_data_frame(
        df_stacked=df_regular_season_playoffs_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max,
        column_name_list_not_first_suffix_not_second_suffix=['game_id', 'game_date', 'season', 'season_type'],
        column_name_indicator='a_b_',
        column_name_suffix_list=['_a', '_b'])


    #get differences
    df_regular_season_playoffs_future_away_home_empty_game_count_day_window_max_diff_future_away_home_game_percentage_day_window_max_diff = \
    get_data_frame_numeric_difference_basic(
        df_two_suffix=df_regular_season_playoffs_future_away_home_empty_game_count_day_window_max_future_away_home_game_percentage_day_window_max_a_b,
        column_name_list_not_difference=['game_id', 'game_date','team_id_a', 'team_id_b'], 
        column_name_suffix_list=['_a', '_b'])

    return df_regular_season_playoffs_future_away_home_empty_game_count_day_window_max_diff_future_away_home_game_percentage_day_window_max_diff


##########################################################################################################################   


    
    
    
    
    


'''
Validation functions
'''

def same_numerics_check(df0, 
                        df1,
                        column_name_list_sort_by=None):
    
    column_name_list_df0_numerics_common_df1_numerics = \
    [k for k in df0.select_dtypes('number').columns if k in df1.columns]
    
    df0_numerics = \
    df0.loc[:, column_name_list_df0_numerics_common_df1_numerics]
   
    df1_numerics = \
    df1.loc[:, column_name_list_df0_numerics_common_df1_numerics]
    
    
    if column_name_list_sort_by == None:
        
        #return descriptive statistics of data frame entry differences
        return (df0_numerics - df1_numerics.values).describe()
    
    elif column_name_list_sort_by != None:
        #sort rows
        df0 = \
        df0.sort_values(column_name_list_sort_by).reset_index(drop=True)
       
        df1 = \
        df1.sort_values(column_name_list_sort_by).reset_index(drop=True)
        
        #return the difference descriptive statistics
        return (df0_numerics - df1_numerics.values).describe()





    
    
'''
Preprocessing Data functions
'''

def get_column_name_dict_collection():
    
    column_name_dict_collection = {}

    column_name_dict_collection['score_difference_a'] = \
    {'score_difference_a':'spread_a'}

    column_name_dict_collection['strength_of_schedule'] = \
    {'win_percentage_diff': 'win_pct_diff',
     'opponent_win_percentage_diff': 'opp_win_pct_diff',
     'opponent_losses_diff': 'opp_losses_diff',
     'opponent_wins_diff': 'opp_wins_diff',
     'opponents_opponents_losses_diff': 'opp_opps_losses_diff',
     'opponents_opponents_wins_diff': 'opp_opps_wins_diff',
     'opponents_opponents_win_percentage_diff': 'opp_opps_win_pct_diff'}

    column_name_dict_collection['lost_contribution'] = \
    {'lc_sum_MIN_season_cma5_diff': 'lc_sum_MIN_cma5_diff',
     'lc_sum_E_OFF_RATING_season_cma5_diff': 'lc_sum_E_OFF_RATING_cma5_diff',
     'lc_sum_OFF_RATING_season_cma5_diff': 'lc_sum_OFF_RATING_cma5_diff',
     'lc_sum_E_DEF_RATING_season_cma5_diff': 'lc_sum_E_DEF_RATING_cma5_diff',
     'lc_sum_DEF_RATING_season_cma5_diff': 'lc_sum_DEF_RATING_cma5_diff',
     'lc_sum_E_NET_RATING_season_cma5_diff': 'lc_sum_E_NET_RATING_cma5_diff',
     'lc_sum_NET_RATING_season_cma5_diff': 'lc_sum_NET_RATING_cma5_diff',
     'lc_sum_AST_PCT_season_cma5_diff': 'lc_sum_AST_PCT_cma5_diff',
     'lc_sum_AST_TOV_season_cma5_diff': 'lc_sum_AST_TOV_cma5_diff',
     'lc_sum_AST_RATIO_season_cma5_diff': 'lc_sum_AST_RATIO_cma5_diff',
     'lc_sum_OREB_PCT_season_cma5_diff': 'lc_sum_OREB_PCT_cma5_diff',
     'lc_sum_DREB_PCT_season_cma5_diff': 'lc_sum_DREB_PCT_cma5_diff',
     'lc_sum_REB_PCT_season_cma5_diff': 'lc_sum_REB_PCT_cma5_diff',
     'lc_sum_TM_TOV_PCT_season_cma5_diff': 'lc_sum_TM_TOV_PCT_cma5_diff',
     'lc_sum_EFG_PCT_season_cma5_diff': 'lc_sum_EFG_PCT_cma5_diff',
     'lc_sum_TS_PCT_season_cma5_diff': 'lc_sum_TS_PCT_cma5_diff',
     'lc_sum_USG_PCT_season_cma5_diff': 'lc_sum_USG_PCT_cma5_diff',
     'lc_sum_E_USG_PCT_season_cma5_diff': 'lc_sum_E_USG_PCT_cma5_diff',
     'lc_sum_E_PACE_season_cma5_diff': 'lc_sum_E_PACE_cma5_diff',
     'lc_sum_PACE_season_cma5_diff': 'lc_sum_PACE_cma5_diff',
     'lc_sum_PACE_PER40_season_cma5_diff': 'lc_sum_PACE_PER40_cma5_diff',
     'lc_sum_POSS_season_cma5_diff': 'lc_sum_POSS_cma5_diff',
     'lc_sum_PIE_season_cma5_diff': 'lc_sum_PIE_cma5_diff',
     'lc_mean_MIN_season_cma5_diff': 'lc_mean_MIN_cma5_diff',
     'lc_mean_E_OFF_RATING_season_cma5_diff': 'lc_mean_E_OFF_RATING_cma5_diff',
     'lc_mean_OFF_RATING_season_cma5_diff': 'lc_mean_OFF_RATING_cma5_diff',
     'lc_mean_E_DEF_RATING_season_cma5_diff': 'lc_mean_E_DEF_RATING_cma5_diff',
     'lc_mean_DEF_RATING_season_cma5_diff': 'lc_mean_DEF_RATING_cma5_diff',
     'lc_mean_E_NET_RATING_season_cma5_diff': 'lc_mean_E_NET_RATING_cma5_diff',
     'lc_mean_NET_RATING_season_cma5_diff': 'lc_mean_NET_RATING_cma5_diff',
     'lc_mean_AST_PCT_season_cma5_diff': 'lc_mean_AST_PCT_cma5_diff',
     'lc_mean_AST_TOV_season_cma5_diff': 'lc_mean_AST_TOV_cma5_diff',
     'lc_mean_AST_RATIO_season_cma5_diff': 'lc_mean_AST_RATIO_cma5_diff',
     'lc_mean_OREB_PCT_season_cma5_diff': 'lc_mean_OREB_PCT_cma5_diff',
     'lc_mean_DREB_PCT_season_cma5_diff': 'lc_mean_DREB_PCT_cma5_diff',
     'lc_mean_REB_PCT_season_cma5_diff': 'lc_mean_REB_PCT_cma5_diff',
     'lc_mean_TM_TOV_PCT_season_cma5_diff': 'lc_mean_TM_TOV_PCT_cma5_diff',
     'lc_mean_EFG_PCT_season_cma5_diff': 'lc_mean_EFG_PCT_cma5_diff',
     'lc_mean_TS_PCT_season_cma5_diff': 'lc_mean_TS_PCT_cma5_diff',
     'lc_mean_USG_PCT_season_cma5_diff': 'lc_mean_USG_PCT_cma5_diff',
     'lc_mean_E_USG_PCT_season_cma5_diff': 'lc_mean_E_USG_PCT_cma5_diff',
     'lc_mean_E_PACE_season_cma5_diff': 'lc_mean_E_PACE_cma5_diff',
     'lc_mean_PACE_season_cma5_diff': 'lc_mean_PACE_cma5_diff',
     'lc_mean_PACE_PER40_season_cma5_diff': 'lc_mean_PACE_PER40_cma5_diff',
     'lc_mean_POSS_season_cma5_diff': 'lc_mean_POSS_cma5_diff',
     'lc_mean_PIE_season_cma5_diff': 'lc_mean_PIE_cma5_diff',
     'lc_max_MIN_season_cma5_diff': 'lc_max_MIN_cma5_diff',
     'lc_max_E_OFF_RATING_season_cma5_diff': 'lc_max_E_OFF_RATING_cma5_diff',
     'lc_max_OFF_RATING_season_cma5_diff': 'lc_max_OFF_RATING_cma5_diff',
     'lc_max_E_DEF_RATING_season_cma5_diff': 'lc_max_E_DEF_RATING_cma5_diff',
     'lc_max_DEF_RATING_season_cma5_diff': 'lc_max_DEF_RATING_cma5_diff',
     'lc_max_E_NET_RATING_season_cma5_diff': 'lc_max_E_NET_RATING_cma5_diff',
     'lc_max_NET_RATING_season_cma5_diff': 'lc_max_NET_RATING_cma5_diff',
     'lc_max_AST_PCT_season_cma5_diff': 'lc_max_AST_PCT_cma5_diff',
     'lc_max_AST_TOV_season_cma5_diff': 'lc_max_AST_TOV_cma5_diff',
     'lc_max_AST_RATIO_season_cma5_diff': 'lc_max_AST_RATIO_cma5_diff',
     'lc_max_OREB_PCT_season_cma5_diff': 'lc_max_OREB_PCT_cma5_diff',
     'lc_max_DREB_PCT_season_cma5_diff': 'lc_max_DREB_PCT_cma5_diff',
     'lc_max_REB_PCT_season_cma5_diff': 'lc_max_REB_PCT_cma5_diff',
     'lc_max_TM_TOV_PCT_season_cma5_diff': 'lc_max_TM_TOV_PCT_cma5_diff',
     'lc_max_EFG_PCT_season_cma5_diff': 'lc_max_EFG_PCT_cma5_diff',
     'lc_max_TS_PCT_season_cma5_diff': 'lc_max_TS_PCT_cma5_diff',
     'lc_max_USG_PCT_season_cma5_diff': 'lc_max_USG_PCT_cma5_diff',
     'lc_max_E_USG_PCT_season_cma5_diff': 'lc_max_E_USG_PCT_cma5_diff',
     'lc_max_E_PACE_season_cma5_diff': 'lc_max_E_PACE_cma5_diff',
     'lc_max_PACE_season_cma5_diff': 'lc_max_PACE_cma5_diff',
     'lc_max_PACE_PER40_season_cma5_diff': 'lc_max_PACE_PER40_cma5_diff',
     'lc_max_POSS_season_cma5_diff': 'lc_max_POSS_cma5_diff',
     'lc_max_PIE_season_cma5_diff': 'lc_max_PIE_cma5_diff',
     'lc_sum_MIN_season_cma12_diff': 'lc_sum_MIN_cma12_diff',
     'lc_sum_E_OFF_RATING_season_cma12_diff': 'lc_sum_E_OFF_RATING_cma12_diff',
     'lc_sum_OFF_RATING_season_cma12_diff': 'lc_sum_OFF_RATING_cma12_diff',
     'lc_sum_E_DEF_RATING_season_cma12_diff': 'lc_sum_E_DEF_RATING_cma12_diff',
     'lc_sum_DEF_RATING_season_cma12_diff': 'lc_sum_DEF_RATING_cma12_diff',
     'lc_sum_E_NET_RATING_season_cma12_diff': 'lc_sum_E_NET_RATING_cma12_diff',
     'lc_sum_NET_RATING_season_cma12_diff': 'lc_sum_NET_RATING_cma12_diff',
     'lc_sum_AST_PCT_season_cma12_diff': 'lc_sum_AST_PCT_cma12_diff',
     'lc_sum_AST_TOV_season_cma12_diff': 'lc_sum_AST_TOV_cma12_diff',
     'lc_sum_AST_RATIO_season_cma12_diff': 'lc_sum_AST_RATIO_cma12_diff',
     'lc_sum_OREB_PCT_season_cma12_diff': 'lc_sum_OREB_PCT_cma12_diff',
     'lc_sum_DREB_PCT_season_cma12_diff': 'lc_sum_DREB_PCT_cma12_diff',
     'lc_sum_REB_PCT_season_cma12_diff': 'lc_sum_REB_PCT_cma12_diff',
     'lc_sum_TM_TOV_PCT_season_cma12_diff': 'lc_sum_TM_TOV_PCT_cma12_diff',
     'lc_sum_EFG_PCT_season_cma12_diff': 'lc_sum_EFG_PCT_cma12_diff',
     'lc_sum_TS_PCT_season_cma12_diff': 'lc_sum_TS_PCT_cma12_diff',
     'lc_sum_USG_PCT_season_cma12_diff': 'lc_sum_USG_PCT_cma12_diff',
     'lc_sum_E_USG_PCT_season_cma12_diff': 'lc_sum_E_USG_PCT_cma12_diff',
     'lc_sum_E_PACE_season_cma12_diff': 'lc_sum_E_PACE_cma12_diff',
     'lc_sum_PACE_season_cma12_diff': 'lc_sum_PACE_cma12_diff',
     'lc_sum_PACE_PER40_season_cma12_diff': 'lc_sum_PACE_PER40_cma12_diff',
     'lc_sum_POSS_season_cma12_diff': 'lc_sum_POSS_cma12_diff',
     'lc_sum_PIE_season_cma12_diff': 'lc_sum_PIE_cma12_diff',
     'lc_mean_MIN_season_cma12_diff': 'lc_mean_MIN_cma12_diff',
     'lc_mean_E_OFF_RATING_season_cma12_diff': 'lc_mean_E_OFF_RATING_cma12_diff',
     'lc_mean_OFF_RATING_season_cma12_diff': 'lc_mean_OFF_RATING_cma12_diff',
     'lc_mean_E_DEF_RATING_season_cma12_diff': 'lc_mean_E_DEF_RATING_cma12_diff',
     'lc_mean_DEF_RATING_season_cma12_diff': 'lc_mean_DEF_RATING_cma12_diff',
     'lc_mean_E_NET_RATING_season_cma12_diff': 'lc_mean_E_NET_RATING_cma12_diff',
     'lc_mean_NET_RATING_season_cma12_diff': 'lc_mean_NET_RATING_cma12_diff',
     'lc_mean_AST_PCT_season_cma12_diff': 'lc_mean_AST_PCT_cma12_diff',
     'lc_mean_AST_TOV_season_cma12_diff': 'lc_mean_AST_TOV_cma12_diff',
     'lc_mean_AST_RATIO_season_cma12_diff': 'lc_mean_AST_RATIO_cma12_diff',
     'lc_mean_OREB_PCT_season_cma12_diff': 'lc_mean_OREB_PCT_cma12_diff',
     'lc_mean_DREB_PCT_season_cma12_diff': 'lc_mean_DREB_PCT_cma12_diff',
     'lc_mean_REB_PCT_season_cma12_diff': 'lc_mean_REB_PCT_cma12_diff',
     'lc_mean_TM_TOV_PCT_season_cma12_diff': 'lc_mean_TM_TOV_PCT_cma12_diff',
     'lc_mean_EFG_PCT_season_cma12_diff': 'lc_mean_EFG_PCT_cma12_diff',
     'lc_mean_TS_PCT_season_cma12_diff': 'lc_mean_TS_PCT_cma12_diff',
     'lc_mean_USG_PCT_season_cma12_diff': 'lc_mean_USG_PCT_cma12_diff',
     'lc_mean_E_USG_PCT_season_cma12_diff': 'lc_mean_E_USG_PCT_cma12_diff',
     'lc_mean_E_PACE_season_cma12_diff': 'lc_mean_E_PACE_cma12_diff',
     'lc_mean_PACE_season_cma12_diff': 'lc_mean_PACE_cma12_diff',
     'lc_mean_PACE_PER40_season_cma12_diff': 'lc_mean_PACE_PER40_cma12_diff',
     'lc_mean_POSS_season_cma12_diff': 'lc_mean_POSS_cma12_diff',
     'lc_mean_PIE_season_cma12_diff': 'lc_mean_PIE_cma12_diff',
     'lc_max_MIN_season_cma12_diff': 'lc_max_MIN_cma12_diff',
     'lc_max_E_OFF_RATING_season_cma12_diff': 'lc_max_E_OFF_RATING_cma12_diff',
     'lc_max_OFF_RATING_season_cma12_diff': 'lc_max_OFF_RATING_cma12_diff',
     'lc_max_E_DEF_RATING_season_cma12_diff': 'lc_max_E_DEF_RATING_cma12_diff',
     'lc_max_DEF_RATING_season_cma12_diff': 'lc_max_DEF_RATING_cma12_diff',
     'lc_max_E_NET_RATING_season_cma12_diff': 'lc_max_E_NET_RATING_cma12_diff',
     'lc_max_NET_RATING_season_cma12_diff': 'lc_max_NET_RATING_cma12_diff',
     'lc_max_AST_PCT_season_cma12_diff': 'lc_max_AST_PCT_cma12_diff',
     'lc_max_AST_TOV_season_cma12_diff': 'lc_max_AST_TOV_cma12_diff',
     'lc_max_AST_RATIO_season_cma12_diff': 'lc_max_AST_RATIO_cma12_diff',
     'lc_max_OREB_PCT_season_cma12_diff': 'lc_max_OREB_PCT_cma12_diff',
     'lc_max_DREB_PCT_season_cma12_diff': 'lc_max_DREB_PCT_cma12_diff',
     'lc_max_REB_PCT_season_cma12_diff': 'lc_max_REB_PCT_cma12_diff',
     'lc_max_TM_TOV_PCT_season_cma12_diff': 'lc_max_TM_TOV_PCT_cma12_diff',
     'lc_max_EFG_PCT_season_cma12_diff': 'lc_max_EFG_PCT_cma12_diff',
     'lc_max_TS_PCT_season_cma12_diff': 'lc_max_TS_PCT_cma12_diff',
     'lc_max_USG_PCT_season_cma12_diff': 'lc_max_USG_PCT_cma12_diff',
     'lc_max_E_USG_PCT_season_cma12_diff': 'lc_max_E_USG_PCT_cma12_diff',
     'lc_max_E_PACE_season_cma12_diff': 'lc_max_E_PACE_cma12_diff',
     'lc_max_PACE_season_cma12_diff': 'lc_max_PACE_cma12_diff',
     'lc_max_PACE_PER40_season_cma12_diff': 'lc_max_PACE_PER40_cma12_diff',
     'lc_max_POSS_season_cma12_diff': 'lc_max_POSS_cma12_diff',
     'lc_max_PIE_season_cma12_diff': 'lc_max_PIE_cma12_diff',
     'lc_sum_MIN_season_cma999_diff': 'lc_sum_MIN_cma999_diff',
     'lc_sum_E_OFF_RATING_season_cma999_diff': 'lc_sum_E_OFF_RATING_cma999_diff',
     'lc_sum_OFF_RATING_season_cma999_diff': 'lc_sum_OFF_RATING_cma999_diff',
     'lc_sum_E_DEF_RATING_season_cma999_diff': 'lc_sum_E_DEF_RATING_cma999_diff',
     'lc_sum_DEF_RATING_season_cma999_diff': 'lc_sum_DEF_RATING_cma999_diff',
     'lc_sum_E_NET_RATING_season_cma999_diff': 'lc_sum_E_NET_RATING_cma999_diff',
     'lc_sum_NET_RATING_season_cma999_diff': 'lc_sum_NET_RATING_cma999_diff',
     'lc_sum_AST_PCT_season_cma999_diff': 'lc_sum_AST_PCT_cma999_diff',
     'lc_sum_AST_TOV_season_cma999_diff': 'lc_sum_AST_TOV_cma999_diff',
     'lc_sum_AST_RATIO_season_cma999_diff': 'lc_sum_AST_RATIO_cma999_diff',
     'lc_sum_OREB_PCT_season_cma999_diff': 'lc_sum_OREB_PCT_cma999_diff',
     'lc_sum_DREB_PCT_season_cma999_diff': 'lc_sum_DREB_PCT_cma999_diff',
     'lc_sum_REB_PCT_season_cma999_diff': 'lc_sum_REB_PCT_cma999_diff',
     'lc_sum_TM_TOV_PCT_season_cma999_diff': 'lc_sum_TM_TOV_PCT_cma999_diff',
     'lc_sum_EFG_PCT_season_cma999_diff': 'lc_sum_EFG_PCT_cma999_diff',
     'lc_sum_TS_PCT_season_cma999_diff': 'lc_sum_TS_PCT_cma999_diff',
     'lc_sum_USG_PCT_season_cma999_diff': 'lc_sum_USG_PCT_cma999_diff',
     'lc_sum_E_USG_PCT_season_cma999_diff': 'lc_sum_E_USG_PCT_cma999_diff',
     'lc_sum_E_PACE_season_cma999_diff': 'lc_sum_E_PACE_cma999_diff',
     'lc_sum_PACE_season_cma999_diff': 'lc_sum_PACE_cma999_diff',
     'lc_sum_PACE_PER40_season_cma999_diff': 'lc_sum_PACE_PER40_cma999_diff',
     'lc_sum_POSS_season_cma999_diff': 'lc_sum_POSS_cma999_diff',
     'lc_sum_PIE_season_cma999_diff': 'lc_sum_PIE_cma999_diff',
     'lc_mean_MIN_season_cma999_diff': 'lc_mean_MIN_cma999_diff',
     'lc_mean_E_OFF_RATING_season_cma999_diff': 'lc_mean_E_OFF_RATING_cma999_diff',
     'lc_mean_OFF_RATING_season_cma999_diff': 'lc_mean_OFF_RATING_cma999_diff',
     'lc_mean_E_DEF_RATING_season_cma999_diff': 'lc_mean_E_DEF_RATING_cma999_diff',
     'lc_mean_DEF_RATING_season_cma999_diff': 'lc_mean_DEF_RATING_cma999_diff',
     'lc_mean_E_NET_RATING_season_cma999_diff': 'lc_mean_E_NET_RATING_cma999_diff',
     'lc_mean_NET_RATING_season_cma999_diff': 'lc_mean_NET_RATING_cma999_diff',
     'lc_mean_AST_PCT_season_cma999_diff': 'lc_mean_AST_PCT_cma999_diff',
     'lc_mean_AST_TOV_season_cma999_diff': 'lc_mean_AST_TOV_cma999_diff',
     'lc_mean_AST_RATIO_season_cma999_diff': 'lc_mean_AST_RATIO_cma999_diff',
     'lc_mean_OREB_PCT_season_cma999_diff': 'lc_mean_OREB_PCT_cma999_diff',
     'lc_mean_DREB_PCT_season_cma999_diff': 'lc_mean_DREB_PCT_cma999_diff',
     'lc_mean_REB_PCT_season_cma999_diff': 'lc_mean_REB_PCT_cma999_diff',
     'lc_mean_TM_TOV_PCT_season_cma999_diff': 'lc_mean_TM_TOV_PCT_cma999_diff',
     'lc_mean_EFG_PCT_season_cma999_diff': 'lc_mean_EFG_PCT_cma999_diff',
     'lc_mean_TS_PCT_season_cma999_diff': 'lc_mean_TS_PCT_cma999_diff',
     'lc_mean_USG_PCT_season_cma999_diff': 'lc_mean_USG_PCT_cma999_diff',
     'lc_mean_E_USG_PCT_season_cma999_diff': 'lc_mean_E_USG_PCT_cma999_diff',
     'lc_mean_E_PACE_season_cma999_diff': 'lc_mean_E_PACE_cma999_diff',
     'lc_mean_PACE_season_cma999_diff': 'lc_mean_PACE_cma999_diff',
     'lc_mean_PACE_PER40_season_cma999_diff': 'lc_mean_PACE_PER40_cma999_diff',
     'lc_mean_POSS_season_cma999_diff': 'lc_mean_POSS_cma999_diff',
     'lc_mean_PIE_season_cma999_diff': 'lc_mean_PIE_cma999_diff',
     'lc_max_MIN_season_cma999_diff': 'lc_max_MIN_cma999_diff',
     'lc_max_E_OFF_RATING_season_cma999_diff': 'lc_max_E_OFF_RATING_cma999_diff',
     'lc_max_OFF_RATING_season_cma999_diff': 'lc_max_OFF_RATING_cma999_diff',
     'lc_max_E_DEF_RATING_season_cma999_diff': 'lc_max_E_DEF_RATING_cma999_diff',
     'lc_max_DEF_RATING_season_cma999_diff': 'lc_max_DEF_RATING_cma999_diff',
     'lc_max_E_NET_RATING_season_cma999_diff': 'lc_max_E_NET_RATING_cma999_diff',
     'lc_max_NET_RATING_season_cma999_diff': 'lc_max_NET_RATING_cma999_diff',
     'lc_max_AST_PCT_season_cma999_diff': 'lc_max_AST_PCT_cma999_diff',
     'lc_max_AST_TOV_season_cma999_diff': 'lc_max_AST_TOV_cma999_diff',
     'lc_max_AST_RATIO_season_cma999_diff': 'lc_max_AST_RATIO_cma999_diff',
     'lc_max_OREB_PCT_season_cma999_diff': 'lc_max_OREB_PCT_cma999_diff',
     'lc_max_DREB_PCT_season_cma999_diff': 'lc_max_DREB_PCT_cma999_diff',
     'lc_max_REB_PCT_season_cma999_diff': 'lc_max_REB_PCT_cma999_diff',
     'lc_max_TM_TOV_PCT_season_cma999_diff': 'lc_max_TM_TOV_PCT_cma999_diff',
     'lc_max_EFG_PCT_season_cma999_diff': 'lc_max_EFG_PCT_cma999_diff',
     'lc_max_TS_PCT_season_cma999_diff': 'lc_max_TS_PCT_cma999_diff',
     'lc_max_USG_PCT_season_cma999_diff': 'lc_max_USG_PCT_cma999_diff',
     'lc_max_E_USG_PCT_season_cma999_diff': 'lc_max_E_USG_PCT_cma999_diff',
     'lc_max_E_PACE_season_cma999_diff': 'lc_max_E_PACE_cma999_diff',
     'lc_max_PACE_season_cma999_diff': 'lc_max_PACE_cma999_diff',
     'lc_max_PACE_PER40_season_cma999_diff': 'lc_max_PACE_PER40_cma999_diff',
     'lc_max_POSS_season_cma999_diff': 'lc_max_POSS_cma999_diff',
     'lc_max_PIE_season_cma999_diff': 'lc_max_PIE_cma999_diff',
     'lc_sum_MIN_season_cmax5_diff': 'lc_sum_MIN_cmax5_diff',
     'lc_sum_E_OFF_RATING_season_cmax5_diff': 'lc_sum_E_OFF_RATING_cmax5_diff',
     'lc_sum_OFF_RATING_season_cmax5_diff': 'lc_sum_OFF_RATING_cmax5_diff',
     'lc_sum_E_DEF_RATING_season_cmax5_diff': 'lc_sum_E_DEF_RATING_cmax5_diff',
     'lc_sum_DEF_RATING_season_cmax5_diff': 'lc_sum_DEF_RATING_cmax5_diff',
     'lc_sum_E_NET_RATING_season_cmax5_diff': 'lc_sum_E_NET_RATING_cmax5_diff',
     'lc_sum_NET_RATING_season_cmax5_diff': 'lc_sum_NET_RATING_cmax5_diff',
     'lc_sum_AST_PCT_season_cmax5_diff': 'lc_sum_AST_PCT_cmax5_diff',
     'lc_sum_AST_TOV_season_cmax5_diff': 'lc_sum_AST_TOV_cmax5_diff',
     'lc_sum_AST_RATIO_season_cmax5_diff': 'lc_sum_AST_RATIO_cmax5_diff',
     'lc_sum_OREB_PCT_season_cmax5_diff': 'lc_sum_OREB_PCT_cmax5_diff',
     'lc_sum_DREB_PCT_season_cmax5_diff': 'lc_sum_DREB_PCT_cmax5_diff',
     'lc_sum_REB_PCT_season_cmax5_diff': 'lc_sum_REB_PCT_cmax5_diff',
     'lc_sum_TM_TOV_PCT_season_cmax5_diff': 'lc_sum_TM_TOV_PCT_cmax5_diff',
     'lc_sum_EFG_PCT_season_cmax5_diff': 'lc_sum_EFG_PCT_cmax5_diff',
     'lc_sum_TS_PCT_season_cmax5_diff': 'lc_sum_TS_PCT_cmax5_diff',
     'lc_sum_USG_PCT_season_cmax5_diff': 'lc_sum_USG_PCT_cmax5_diff',
     'lc_sum_E_USG_PCT_season_cmax5_diff': 'lc_sum_E_USG_PCT_cmax5_diff',
     'lc_sum_E_PACE_season_cmax5_diff': 'lc_sum_E_PACE_cmax5_diff',
     'lc_sum_PACE_season_cmax5_diff': 'lc_sum_PACE_cmax5_diff',
     'lc_sum_PACE_PER40_season_cmax5_diff': 'lc_sum_PACE_PER40_cmax5_diff',
     'lc_sum_POSS_season_cmax5_diff': 'lc_sum_POSS_cmax5_diff',
     'lc_sum_PIE_season_cmax5_diff': 'lc_sum_PIE_cmax5_diff',
     'lc_mean_MIN_season_cmax5_diff': 'lc_mean_MIN_cmax5_diff',
     'lc_mean_E_OFF_RATING_season_cmax5_diff': 'lc_mean_E_OFF_RATING_cmax5_diff',
     'lc_mean_OFF_RATING_season_cmax5_diff': 'lc_mean_OFF_RATING_cmax5_diff',
     'lc_mean_E_DEF_RATING_season_cmax5_diff': 'lc_mean_E_DEF_RATING_cmax5_diff',
     'lc_mean_DEF_RATING_season_cmax5_diff': 'lc_mean_DEF_RATING_cmax5_diff',
     'lc_mean_E_NET_RATING_season_cmax5_diff': 'lc_mean_E_NET_RATING_cmax5_diff',
     'lc_mean_NET_RATING_season_cmax5_diff': 'lc_mean_NET_RATING_cmax5_diff',
     'lc_mean_AST_PCT_season_cmax5_diff': 'lc_mean_AST_PCT_cmax5_diff',
     'lc_mean_AST_TOV_season_cmax5_diff': 'lc_mean_AST_TOV_cmax5_diff',
     'lc_mean_AST_RATIO_season_cmax5_diff': 'lc_mean_AST_RATIO_cmax5_diff',
     'lc_mean_OREB_PCT_season_cmax5_diff': 'lc_mean_OREB_PCT_cmax5_diff',
     'lc_mean_DREB_PCT_season_cmax5_diff': 'lc_mean_DREB_PCT_cmax5_diff',
     'lc_mean_REB_PCT_season_cmax5_diff': 'lc_mean_REB_PCT_cmax5_diff',
     'lc_mean_TM_TOV_PCT_season_cmax5_diff': 'lc_mean_TM_TOV_PCT_cmax5_diff',
     'lc_mean_EFG_PCT_season_cmax5_diff': 'lc_mean_EFG_PCT_cmax5_diff',
     'lc_mean_TS_PCT_season_cmax5_diff': 'lc_mean_TS_PCT_cmax5_diff',
     'lc_mean_USG_PCT_season_cmax5_diff': 'lc_mean_USG_PCT_cmax5_diff',
     'lc_mean_E_USG_PCT_season_cmax5_diff': 'lc_mean_E_USG_PCT_cmax5_diff',
     'lc_mean_E_PACE_season_cmax5_diff': 'lc_mean_E_PACE_cmax5_diff',
     'lc_mean_PACE_season_cmax5_diff': 'lc_mean_PACE_cmax5_diff',
     'lc_mean_PACE_PER40_season_cmax5_diff': 'lc_mean_PACE_PER40_cmax5_diff',
     'lc_mean_POSS_season_cmax5_diff': 'lc_mean_POSS_cmax5_diff',
     'lc_mean_PIE_season_cmax5_diff': 'lc_mean_PIE_cmax5_diff',
     'lc_max_MIN_season_cmax5_diff': 'lc_max_MIN_cmax5_diff',
     'lc_max_E_OFF_RATING_season_cmax5_diff': 'lc_max_E_OFF_RATING_cmax5_diff',
     'lc_max_OFF_RATING_season_cmax5_diff': 'lc_max_OFF_RATING_cmax5_diff',
     'lc_max_E_DEF_RATING_season_cmax5_diff': 'lc_max_E_DEF_RATING_cmax5_diff',
     'lc_max_DEF_RATING_season_cmax5_diff': 'lc_max_DEF_RATING_cmax5_diff',
     'lc_max_E_NET_RATING_season_cmax5_diff': 'lc_max_E_NET_RATING_cmax5_diff',
     'lc_max_NET_RATING_season_cmax5_diff': 'lc_max_NET_RATING_cmax5_diff',
     'lc_max_AST_PCT_season_cmax5_diff': 'lc_max_AST_PCT_cmax5_diff',
     'lc_max_AST_TOV_season_cmax5_diff': 'lc_max_AST_TOV_cmax5_diff',
     'lc_max_AST_RATIO_season_cmax5_diff': 'lc_max_AST_RATIO_cmax5_diff',
     'lc_max_OREB_PCT_season_cmax5_diff': 'lc_max_OREB_PCT_cmax5_diff',
     'lc_max_DREB_PCT_season_cmax5_diff': 'lc_max_DREB_PCT_cmax5_diff',
     'lc_max_REB_PCT_season_cmax5_diff': 'lc_max_REB_PCT_cmax5_diff',
     'lc_max_TM_TOV_PCT_season_cmax5_diff': 'lc_max_TM_TOV_PCT_cmax5_diff',
     'lc_max_EFG_PCT_season_cmax5_diff': 'lc_max_EFG_PCT_cmax5_diff',
     'lc_max_TS_PCT_season_cmax5_diff': 'lc_max_TS_PCT_cmax5_diff',
     'lc_max_USG_PCT_season_cmax5_diff': 'lc_max_USG_PCT_cmax5_diff',
     'lc_max_E_USG_PCT_season_cmax5_diff': 'lc_max_E_USG_PCT_cmax5_diff',
     'lc_max_E_PACE_season_cmax5_diff': 'lc_max_E_PACE_cmax5_diff',
     'lc_max_PACE_season_cmax5_diff': 'lc_max_PACE_cmax5_diff',
     'lc_max_PACE_PER40_season_cmax5_diff': 'lc_max_PACE_PER40_cmax5_diff',
     'lc_max_POSS_season_cmax5_diff': 'lc_max_POSS_cmax5_diff',
     'lc_max_PIE_season_cmax5_diff': 'lc_max_PIE_cmax5_diff',
     'lc_sum_MIN_season_cmax12_diff': 'lc_sum_MIN_cmax12_diff',
     'lc_sum_E_OFF_RATING_season_cmax12_diff': 'lc_sum_E_OFF_RATING_cmax12_diff',
     'lc_sum_OFF_RATING_season_cmax12_diff': 'lc_sum_OFF_RATING_cmax12_diff',
     'lc_sum_E_DEF_RATING_season_cmax12_diff': 'lc_sum_E_DEF_RATING_cmax12_diff',
     'lc_sum_DEF_RATING_season_cmax12_diff': 'lc_sum_DEF_RATING_cmax12_diff',
     'lc_sum_E_NET_RATING_season_cmax12_diff': 'lc_sum_E_NET_RATING_cmax12_diff',
     'lc_sum_NET_RATING_season_cmax12_diff': 'lc_sum_NET_RATING_cmax12_diff',
     'lc_sum_AST_PCT_season_cmax12_diff': 'lc_sum_AST_PCT_cmax12_diff',
     'lc_sum_AST_TOV_season_cmax12_diff': 'lc_sum_AST_TOV_cmax12_diff',
     'lc_sum_AST_RATIO_season_cmax12_diff': 'lc_sum_AST_RATIO_cmax12_diff',
     'lc_sum_OREB_PCT_season_cmax12_diff': 'lc_sum_OREB_PCT_cmax12_diff',
     'lc_sum_DREB_PCT_season_cmax12_diff': 'lc_sum_DREB_PCT_cmax12_diff',
     'lc_sum_REB_PCT_season_cmax12_diff': 'lc_sum_REB_PCT_cmax12_diff',
     'lc_sum_TM_TOV_PCT_season_cmax12_diff': 'lc_sum_TM_TOV_PCT_cmax12_diff',
     'lc_sum_EFG_PCT_season_cmax12_diff': 'lc_sum_EFG_PCT_cmax12_diff',
     'lc_sum_TS_PCT_season_cmax12_diff': 'lc_sum_TS_PCT_cmax12_diff',
     'lc_sum_USG_PCT_season_cmax12_diff': 'lc_sum_USG_PCT_cmax12_diff',
     'lc_sum_E_USG_PCT_season_cmax12_diff': 'lc_sum_E_USG_PCT_cmax12_diff',
     'lc_sum_E_PACE_season_cmax12_diff': 'lc_sum_E_PACE_cmax12_diff',
     'lc_sum_PACE_season_cmax12_diff': 'lc_sum_PACE_cmax12_diff',
     'lc_sum_PACE_PER40_season_cmax12_diff': 'lc_sum_PACE_PER40_cmax12_diff',
     'lc_sum_POSS_season_cmax12_diff': 'lc_sum_POSS_cmax12_diff',
     'lc_sum_PIE_season_cmax12_diff': 'lc_sum_PIE_cmax12_diff',
     'lc_mean_MIN_season_cmax12_diff': 'lc_mean_MIN_cmax12_diff',
     'lc_mean_E_OFF_RATING_season_cmax12_diff': 'lc_mean_E_OFF_RATING_cmax12_diff',
     'lc_mean_OFF_RATING_season_cmax12_diff': 'lc_mean_OFF_RATING_cmax12_diff',
     'lc_mean_E_DEF_RATING_season_cmax12_diff': 'lc_mean_E_DEF_RATING_cmax12_diff',
     'lc_mean_DEF_RATING_season_cmax12_diff': 'lc_mean_DEF_RATING_cmax12_diff',
     'lc_mean_E_NET_RATING_season_cmax12_diff': 'lc_mean_E_NET_RATING_cmax12_diff',
     'lc_mean_NET_RATING_season_cmax12_diff': 'lc_mean_NET_RATING_cmax12_diff',
     'lc_mean_AST_PCT_season_cmax12_diff': 'lc_mean_AST_PCT_cmax12_diff',
     'lc_mean_AST_TOV_season_cmax12_diff': 'lc_mean_AST_TOV_cmax12_diff',
     'lc_mean_AST_RATIO_season_cmax12_diff': 'lc_mean_AST_RATIO_cmax12_diff',
     'lc_mean_OREB_PCT_season_cmax12_diff': 'lc_mean_OREB_PCT_cmax12_diff',
     'lc_mean_DREB_PCT_season_cmax12_diff': 'lc_mean_DREB_PCT_cmax12_diff',
     'lc_mean_REB_PCT_season_cmax12_diff': 'lc_mean_REB_PCT_cmax12_diff',
     'lc_mean_TM_TOV_PCT_season_cmax12_diff': 'lc_mean_TM_TOV_PCT_cmax12_diff',
     'lc_mean_EFG_PCT_season_cmax12_diff': 'lc_mean_EFG_PCT_cmax12_diff',
     'lc_mean_TS_PCT_season_cmax12_diff': 'lc_mean_TS_PCT_cmax12_diff',
     'lc_mean_USG_PCT_season_cmax12_diff': 'lc_mean_USG_PCT_cmax12_diff',
     'lc_mean_E_USG_PCT_season_cmax12_diff': 'lc_mean_E_USG_PCT_cmax12_diff',
     'lc_mean_E_PACE_season_cmax12_diff': 'lc_mean_E_PACE_cmax12_diff',
     'lc_mean_PACE_season_cmax12_diff': 'lc_mean_PACE_cmax12_diff',
     'lc_mean_PACE_PER40_season_cmax12_diff': 'lc_mean_PACE_PER40_cmax12_diff',
     'lc_mean_POSS_season_cmax12_diff': 'lc_mean_POSS_cmax12_diff',
     'lc_mean_PIE_season_cmax12_diff': 'lc_mean_PIE_cmax12_diff',
     'lc_max_MIN_season_cmax12_diff': 'lc_max_MIN_cmax12_diff',
     'lc_max_E_OFF_RATING_season_cmax12_diff': 'lc_max_E_OFF_RATING_cmax12_diff',
     'lc_max_OFF_RATING_season_cmax12_diff': 'lc_max_OFF_RATING_cmax12_diff',
     'lc_max_E_DEF_RATING_season_cmax12_diff': 'lc_max_E_DEF_RATING_cmax12_diff',
     'lc_max_DEF_RATING_season_cmax12_diff': 'lc_max_DEF_RATING_cmax12_diff',
     'lc_max_E_NET_RATING_season_cmax12_diff': 'lc_max_E_NET_RATING_cmax12_diff',
     'lc_max_NET_RATING_season_cmax12_diff': 'lc_max_NET_RATING_cmax12_diff',
     'lc_max_AST_PCT_season_cmax12_diff': 'lc_max_AST_PCT_cmax12_diff',
     'lc_max_AST_TOV_season_cmax12_diff': 'lc_max_AST_TOV_cmax12_diff',
     'lc_max_AST_RATIO_season_cmax12_diff': 'lc_max_AST_RATIO_cmax12_diff',
     'lc_max_OREB_PCT_season_cmax12_diff': 'lc_max_OREB_PCT_cmax12_diff',
     'lc_max_DREB_PCT_season_cmax12_diff': 'lc_max_DREB_PCT_cmax12_diff',
     'lc_max_REB_PCT_season_cmax12_diff': 'lc_max_REB_PCT_cmax12_diff',
     'lc_max_TM_TOV_PCT_season_cmax12_diff': 'lc_max_TM_TOV_PCT_cmax12_diff',
     'lc_max_EFG_PCT_season_cmax12_diff': 'lc_max_EFG_PCT_cmax12_diff',
     'lc_max_TS_PCT_season_cmax12_diff': 'lc_max_TS_PCT_cmax12_diff',
     'lc_max_USG_PCT_season_cmax12_diff': 'lc_max_USG_PCT_cmax12_diff',
     'lc_max_E_USG_PCT_season_cmax12_diff': 'lc_max_E_USG_PCT_cmax12_diff',
     'lc_max_E_PACE_season_cmax12_diff': 'lc_max_E_PACE_cmax12_diff',
     'lc_max_PACE_season_cmax12_diff': 'lc_max_PACE_cmax12_diff',
     'lc_max_PACE_PER40_season_cmax12_diff': 'lc_max_PACE_PER40_cmax12_diff',
     'lc_max_POSS_season_cmax12_diff': 'lc_max_POSS_cmax12_diff',
     'lc_max_PIE_season_cmax12_diff': 'lc_max_PIE_cmax12_diff',
     'lc_sum_MIN_season_cmax999_diff': 'lc_sum_MIN_cmax999_diff',
     'lc_sum_E_OFF_RATING_season_cmax999_diff': 'lc_sum_E_OFF_RATING_cmax999_diff',
     'lc_sum_OFF_RATING_season_cmax999_diff': 'lc_sum_OFF_RATING_cmax999_diff',
     'lc_sum_E_DEF_RATING_season_cmax999_diff': 'lc_sum_E_DEF_RATING_cmax999_diff',
     'lc_sum_DEF_RATING_season_cmax999_diff': 'lc_sum_DEF_RATING_cmax999_diff',
     'lc_sum_E_NET_RATING_season_cmax999_diff': 'lc_sum_E_NET_RATING_cmax999_diff',
     'lc_sum_NET_RATING_season_cmax999_diff': 'lc_sum_NET_RATING_cmax999_diff',
     'lc_sum_AST_PCT_season_cmax999_diff': 'lc_sum_AST_PCT_cmax999_diff',
     'lc_sum_AST_TOV_season_cmax999_diff': 'lc_sum_AST_TOV_cmax999_diff',
     'lc_sum_AST_RATIO_season_cmax999_diff': 'lc_sum_AST_RATIO_cmax999_diff',
     'lc_sum_OREB_PCT_season_cmax999_diff': 'lc_sum_OREB_PCT_cmax999_diff',
     'lc_sum_DREB_PCT_season_cmax999_diff': 'lc_sum_DREB_PCT_cmax999_diff',
     'lc_sum_REB_PCT_season_cmax999_diff': 'lc_sum_REB_PCT_cmax999_diff',
     'lc_sum_TM_TOV_PCT_season_cmax999_diff': 'lc_sum_TM_TOV_PCT_cmax999_diff',
     'lc_sum_EFG_PCT_season_cmax999_diff': 'lc_sum_EFG_PCT_cmax999_diff',
     'lc_sum_TS_PCT_season_cmax999_diff': 'lc_sum_TS_PCT_cmax999_diff',
     'lc_sum_USG_PCT_season_cmax999_diff': 'lc_sum_USG_PCT_cmax999_diff',
     'lc_sum_E_USG_PCT_season_cmax999_diff': 'lc_sum_E_USG_PCT_cmax999_diff',
     'lc_sum_E_PACE_season_cmax999_diff': 'lc_sum_E_PACE_cmax999_diff',
     'lc_sum_PACE_season_cmax999_diff': 'lc_sum_PACE_cmax999_diff',
     'lc_sum_PACE_PER40_season_cmax999_diff': 'lc_sum_PACE_PER40_cmax999_diff',
     'lc_sum_POSS_season_cmax999_diff': 'lc_sum_POSS_cmax999_diff',
     'lc_sum_PIE_season_cmax999_diff': 'lc_sum_PIE_cmax999_diff',
     'lc_mean_MIN_season_cmax999_diff': 'lc_mean_MIN_cmax999_diff',
     'lc_mean_E_OFF_RATING_season_cmax999_diff': 'lc_mean_E_OFF_RATING_cmax999_diff',
     'lc_mean_OFF_RATING_season_cmax999_diff': 'lc_mean_OFF_RATING_cmax999_diff',
     'lc_mean_E_DEF_RATING_season_cmax999_diff': 'lc_mean_E_DEF_RATING_cmax999_diff',
     'lc_mean_DEF_RATING_season_cmax999_diff': 'lc_mean_DEF_RATING_cmax999_diff',
     'lc_mean_E_NET_RATING_season_cmax999_diff': 'lc_mean_E_NET_RATING_cmax999_diff',
     'lc_mean_NET_RATING_season_cmax999_diff': 'lc_mean_NET_RATING_cmax999_diff',
     'lc_mean_AST_PCT_season_cmax999_diff': 'lc_mean_AST_PCT_cmax999_diff',
     'lc_mean_AST_TOV_season_cmax999_diff': 'lc_mean_AST_TOV_cmax999_diff',
     'lc_mean_AST_RATIO_season_cmax999_diff': 'lc_mean_AST_RATIO_cmax999_diff',
     'lc_mean_OREB_PCT_season_cmax999_diff': 'lc_mean_OREB_PCT_cmax999_diff',
     'lc_mean_DREB_PCT_season_cmax999_diff': 'lc_mean_DREB_PCT_cmax999_diff',
     'lc_mean_REB_PCT_season_cmax999_diff': 'lc_mean_REB_PCT_cmax999_diff',
     'lc_mean_TM_TOV_PCT_season_cmax999_diff': 'lc_mean_TM_TOV_PCT_cmax999_diff',
     'lc_mean_EFG_PCT_season_cmax999_diff': 'lc_mean_EFG_PCT_cmax999_diff',
     'lc_mean_TS_PCT_season_cmax999_diff': 'lc_mean_TS_PCT_cmax999_diff',
     'lc_mean_USG_PCT_season_cmax999_diff': 'lc_mean_USG_PCT_cmax999_diff',
     'lc_mean_E_USG_PCT_season_cmax999_diff': 'lc_mean_E_USG_PCT_cmax999_diff',
     'lc_mean_E_PACE_season_cmax999_diff': 'lc_mean_E_PACE_cmax999_diff',
     'lc_mean_PACE_season_cmax999_diff': 'lc_mean_PACE_cmax999_diff',
     'lc_mean_PACE_PER40_season_cmax999_diff': 'lc_mean_PACE_PER40_cmax999_diff',
     'lc_mean_POSS_season_cmax999_diff': 'lc_mean_POSS_cmax999_diff',
     'lc_mean_PIE_season_cmax999_diff': 'lc_mean_PIE_cmax999_diff',
     'lc_max_MIN_season_cmax999_diff': 'lc_max_MIN_cmax999_diff',
     'lc_max_E_OFF_RATING_season_cmax999_diff': 'lc_max_E_OFF_RATING_cmax999_diff',
     'lc_max_OFF_RATING_season_cmax999_diff': 'lc_max_OFF_RATING_cmax999_diff',
     'lc_max_E_DEF_RATING_season_cmax999_diff': 'lc_max_E_DEF_RATING_cmax999_diff',
     'lc_max_DEF_RATING_season_cmax999_diff': 'lc_max_DEF_RATING_cmax999_diff',
     'lc_max_E_NET_RATING_season_cmax999_diff': 'lc_max_E_NET_RATING_cmax999_diff',
     'lc_max_NET_RATING_season_cmax999_diff': 'lc_max_NET_RATING_cmax999_diff',
     'lc_max_AST_PCT_season_cmax999_diff': 'lc_max_AST_PCT_cmax999_diff',
     'lc_max_AST_TOV_season_cmax999_diff': 'lc_max_AST_TOV_cmax999_diff',
     'lc_max_AST_RATIO_season_cmax999_diff': 'lc_max_AST_RATIO_cmax999_diff',
     'lc_max_OREB_PCT_season_cmax999_diff': 'lc_max_OREB_PCT_cmax999_diff',
     'lc_max_DREB_PCT_season_cmax999_diff': 'lc_max_DREB_PCT_cmax999_diff',
     'lc_max_REB_PCT_season_cmax999_diff': 'lc_max_REB_PCT_cmax999_diff',
     'lc_max_TM_TOV_PCT_season_cmax999_diff': 'lc_max_TM_TOV_PCT_cmax999_diff',
     'lc_max_EFG_PCT_season_cmax999_diff': 'lc_max_EFG_PCT_cmax999_diff',
     'lc_max_TS_PCT_season_cmax999_diff': 'lc_max_TS_PCT_cmax999_diff',
     'lc_max_USG_PCT_season_cmax999_diff': 'lc_max_USG_PCT_cmax999_diff',
     'lc_max_E_USG_PCT_season_cmax999_diff': 'lc_max_E_USG_PCT_cmax999_diff',
     'lc_max_E_PACE_season_cmax999_diff': 'lc_max_E_PACE_cmax999_diff',
     'lc_max_PACE_season_cmax999_diff': 'lc_max_PACE_cmax999_diff',
     'lc_max_PACE_PER40_season_cmax999_diff': 'lc_max_PACE_PER40_cmax999_diff',
     'lc_max_POSS_season_cmax999_diff': 'lc_max_POSS_cmax999_diff',
     'lc_max_PIE_season_cmax999_diff': 'lc_max_PIE_cmax999_diff'}



    column_name_dict_collection['stat'] = \
    {'E_OFF_RATING_season_cma5_diff': 'E_OFF_RATING_cma5_diff',
     'OFF_RATING_season_cma5_diff': 'OFF_RATING_cma5_diff',
     'E_DEF_RATING_season_cma5_diff': 'E_DEF_RATING_cma5_diff',
     'DEF_RATING_season_cma5_diff': 'DEF_RATING_cma5_diff',
     'E_NET_RATING_season_cma5_diff': 'E_NET_RATING_cma5_diff',
     'NET_RATING_season_cma5_diff': 'NET_RATING_cma5_diff',
     'AST_PCT_season_cma5_diff': 'AST_PCT_cma5_diff',
     'AST_TOV_season_cma5_diff': 'AST_TOV_cma5_diff',
     'AST_RATIO_season_cma5_diff': 'AST_RATIO_cma5_diff',
     'OREB_PCT_season_cma5_diff': 'OREB_PCT_cma5_diff',
     'DREB_PCT_season_cma5_diff': 'DREB_PCT_cma5_diff',
     'REB_PCT_season_cma5_diff': 'REB_PCT_cma5_diff',
     'E_TM_TOV_PCT_season_cma5_diff': 'E_TM_TOV_PCT_cma5_diff',
     'TM_TOV_PCT_season_cma5_diff': 'TM_TOV_PCT_cma5_diff',
     'EFG_PCT_season_cma5_diff': 'EFG_PCT_cma5_diff',
     'TS_PCT_season_cma5_diff': 'TS_PCT_cma5_diff',
     'USG_PCT_season_cma5_diff': 'USG_PCT_cma5_diff',
     'E_USG_PCT_season_cma5_diff': 'E_USG_PCT_cma5_diff',
     'E_PACE_season_cma5_diff': 'E_PACE_cma5_diff',
     'PACE_season_cma5_diff': 'PACE_cma5_diff',
     'PACE_PER40_season_cma5_diff': 'PACE_PER40_cma5_diff',
     'POSS_season_cma5_diff': 'POSS_cma5_diff',
     'PIE_season_cma5_diff': 'PIE_cma5_diff',
     'duration_minutes_season_cma5_diff': 'duration_minutes_cma5_diff',
     'w_pct_season_cma5_diff': 'w_pct_cma5_diff',
     'min_season_cma5_diff': 'min_cma5_diff',
     'fgm_season_cma5_diff': 'fgm_cma5_diff',
     'fga_season_cma5_diff': 'fga_cma5_diff',
     'fg_pct_season_cma5_diff': 'fg_pct_cma5_diff',
     'fg3m_season_cma5_diff': 'fg3m_cma5_diff',
     'fg3a_season_cma5_diff': 'fg3a_cma5_diff',
     'fg3_pct_season_cma5_diff': 'fg3_pct_cma5_diff',
     'ftm_season_cma5_diff': 'ftm_cma5_diff',
     'fta_season_cma5_diff': 'fta_cma5_diff',
     'ft_pct_season_cma5_diff': 'ft_pct_cma5_diff',
     'oreb_season_cma5_diff': 'oreb_cma5_diff',
     'dreb_season_cma5_diff': 'dreb_cma5_diff',
     'reb_season_cma5_diff': 'reb_cma5_diff',
     'ast_season_cma5_diff': 'ast_cma5_diff',
     'stl_season_cma5_diff': 'stl_cma5_diff',
     'blk_season_cma5_diff': 'blk_cma5_diff',
     'tov_season_cma5_diff': 'tov_cma5_diff',
     'pf_season_cma5_diff': 'pf_cma5_diff',
     'pts_season_cma5_diff': 'pts_cma5_diff',
     'score_difference_season_cma5_diff': 'spread_cma5_diff',
     'away_home_season_cma5_diff': 'away_home__cma5_diff',
     'E_OFF_RATING_season_cma12_diff': 'E_OFF_RATING_cma12_diff',
     'OFF_RATING_season_cma12_diff': 'OFF_RATING_cma12_diff',
     'E_DEF_RATING_season_cma12_diff': 'E_DEF_RATING_cma12_diff',
     'DEF_RATING_season_cma12_diff': 'DEF_RATING_cma12_diff',
     'E_NET_RATING_season_cma12_diff': 'E_NET_RATING_cma12_diff',
     'NET_RATING_season_cma12_diff': 'NET_RATING_cma12_diff',
     'AST_PCT_season_cma12_diff': 'AST_PCT_cma12_diff',
     'AST_TOV_season_cma12_diff': 'AST_TOV_cma12_diff',
     'AST_RATIO_season_cma12_diff': 'AST_RATIO_cma12_diff',
     'OREB_PCT_season_cma12_diff': 'OREB_PCT_cma12_diff',
     'DREB_PCT_season_cma12_diff': 'DREB_PCT_cma12_diff',
     'REB_PCT_season_cma12_diff': 'REB_PCT_cma12_diff',
     'E_TM_TOV_PCT_season_cma12_diff': 'E_TM_TOV_PCT_cma12_diff',
     'TM_TOV_PCT_season_cma12_diff': 'TM_TOV_PCT_cma12_diff',
     'EFG_PCT_season_cma12_diff': 'EFG_PCT_cma12_diff',
     'TS_PCT_season_cma12_diff': 'TS_PCT_cma12_diff',
     'USG_PCT_season_cma12_diff': 'USG_PCT_cma12_diff',
     'E_USG_PCT_season_cma12_diff': 'E_USG_PCT_cma12_diff',
     'E_PACE_season_cma12_diff': 'E_PACE_cma12_diff',
     'PACE_season_cma12_diff': 'PACE_cma12_diff',
     'PACE_PER40_season_cma12_diff': 'PACE_PER40_cma12_diff',
     'POSS_season_cma12_diff': 'POSS_cma12_diff',
     'PIE_season_cma12_diff': 'PIE_cma12_diff',
     'duration_minutes_season_cma12_diff': 'duration_minutes_cma12_diff',
     'w_pct_season_cma12_diff': 'w_pct_cma12_diff',
     'min_season_cma12_diff': 'min_cma12_diff',
     'fgm_season_cma12_diff': 'fgm_cma12_diff',
     'fga_season_cma12_diff': 'fga_cma12_diff',
     'fg_pct_season_cma12_diff': 'fg_pct_cma12_diff',
     'fg3m_season_cma12_diff': 'fg3m_cma12_diff',
     'fg3a_season_cma12_diff': 'fg3a_cma12_diff',
     'fg3_pct_season_cma12_diff': 'fg3_pct_cma12_diff',
     'ftm_season_cma12_diff': 'ftm_cma12_diff',
     'fta_season_cma12_diff': 'fta_cma12_diff',
     'ft_pct_season_cma12_diff': 'ft_pct_cma12_diff',
     'oreb_season_cma12_diff': 'oreb_cma12_diff',
     'dreb_season_cma12_diff': 'dreb_cma12_diff',
     'reb_season_cma12_diff': 'reb_cma12_diff',
     'ast_season_cma12_diff': 'ast_cma12_diff',
     'stl_season_cma12_diff': 'stl_cma12_diff',
     'blk_season_cma12_diff': 'blk_cma12_diff',
     'tov_season_cma12_diff': 'tov_cma12_diff',
     'pf_season_cma12_diff': 'pf_cma12_diff',
     'pts_season_cma12_diff': 'pts_cma12_diff',
     'score_difference_season_cma12_diff': 'spread_cma12_diff',
     'away_home_season_cma12_diff': 'away_home__cma12_diff',
     'E_OFF_RATING_season_cma999_diff': 'E_OFF_RATING_cma999_diff',
     'OFF_RATING_season_cma999_diff': 'OFF_RATING_cma999_diff',
     'E_DEF_RATING_season_cma999_diff': 'E_DEF_RATING_cma999_diff',
     'DEF_RATING_season_cma999_diff': 'DEF_RATING_cma999_diff',
     'E_NET_RATING_season_cma999_diff': 'E_NET_RATING_cma999_diff',
     'NET_RATING_season_cma999_diff': 'NET_RATING_cma999_diff',
     'AST_PCT_season_cma999_diff': 'AST_PCT_cma999_diff',
     'AST_TOV_season_cma999_diff': 'AST_TOV_cma999_diff',
     'AST_RATIO_season_cma999_diff': 'AST_RATIO_cma999_diff',
     'OREB_PCT_season_cma999_diff': 'OREB_PCT_cma999_diff',
     'DREB_PCT_season_cma999_diff': 'DREB_PCT_cma999_diff',
     'REB_PCT_season_cma999_diff': 'REB_PCT_cma999_diff',
     'E_TM_TOV_PCT_season_cma999_diff': 'E_TM_TOV_PCT_cma999_diff',
     'TM_TOV_PCT_season_cma999_diff': 'TM_TOV_PCT_cma999_diff',
     'EFG_PCT_season_cma999_diff': 'EFG_PCT_cma999_diff',
     'TS_PCT_season_cma999_diff': 'TS_PCT_cma999_diff',
     'USG_PCT_season_cma999_diff': 'USG_PCT_cma999_diff',
     'E_USG_PCT_season_cma999_diff': 'E_USG_PCT_cma999_diff',
     'E_PACE_season_cma999_diff': 'E_PACE_cma999_diff',
     'PACE_season_cma999_diff': 'PACE_cma999_diff',
     'PACE_PER40_season_cma999_diff': 'PACE_PER40_cma999_diff',
     'POSS_season_cma999_diff': 'POSS_cma999_diff',
     'PIE_season_cma999_diff': 'PIE_cma999_diff',
     'duration_minutes_season_cma999_diff': 'duration_minutes_cma999_diff',
     'w_pct_season_cma999_diff': 'w_pct_cma999_diff',
     'min_season_cma999_diff': 'min_cma999_diff',
     'fgm_season_cma999_diff': 'fgm_cma999_diff',
     'fga_season_cma999_diff': 'fga_cma999_diff',
     'fg_pct_season_cma999_diff': 'fg_pct_cma999_diff',
     'fg3m_season_cma999_diff': 'fg3m_cma999_diff',
     'fg3a_season_cma999_diff': 'fg3a_cma999_diff',
     'fg3_pct_season_cma999_diff': 'fg3_pct_cma999_diff',
     'ftm_season_cma999_diff': 'ftm_cma999_diff',
     'fta_season_cma999_diff': 'fta_cma999_diff',
     'ft_pct_season_cma999_diff': 'ft_pct_cma999_diff',
     'oreb_season_cma999_diff': 'oreb_cma999_diff',
     'dreb_season_cma999_diff': 'dreb_cma999_diff',
     'reb_season_cma999_diff': 'reb_cma999_diff',
     'ast_season_cma999_diff': 'ast_cma999_diff',
     'stl_season_cma999_diff': 'stl_cma999_diff',
     'blk_season_cma999_diff': 'blk_cma999_diff',
     'tov_season_cma999_diff': 'tov_cma999_diff',
     'pf_season_cma999_diff': 'pf_cma999_diff',
     'pts_season_cma999_diff': 'pts_cma999_diff',
     'score_difference_season_cma999_diff': 'spread_cma999_diff',
     'away_home_season_cma999_diff': 'away_home__cma999_diff',
     'E_OFF_RATING_season_cmax5_diff': 'E_OFF_RATING_cmax5_diff',
     'OFF_RATING_season_cmax5_diff': 'OFF_RATING_cmax5_diff',
     'E_DEF_RATING_season_cmax5_diff': 'E_DEF_RATING_cmax5_diff',
     'DEF_RATING_season_cmax5_diff': 'DEF_RATING_cmax5_diff',
     'E_NET_RATING_season_cmax5_diff': 'E_NET_RATING_cmax5_diff',
     'NET_RATING_season_cmax5_diff': 'NET_RATING_cmax5_diff',
     'AST_PCT_season_cmax5_diff': 'AST_PCT_cmax5_diff',
     'AST_TOV_season_cmax5_diff': 'AST_TOV_cmax5_diff',
     'AST_RATIO_season_cmax5_diff': 'AST_RATIO_cmax5_diff',
     'OREB_PCT_season_cmax5_diff': 'OREB_PCT_cmax5_diff',
     'DREB_PCT_season_cmax5_diff': 'DREB_PCT_cmax5_diff',
     'REB_PCT_season_cmax5_diff': 'REB_PCT_cmax5_diff',
     'E_TM_TOV_PCT_season_cmax5_diff': 'E_TM_TOV_PCT_cmax5_diff',
     'TM_TOV_PCT_season_cmax5_diff': 'TM_TOV_PCT_cmax5_diff',
     'EFG_PCT_season_cmax5_diff': 'EFG_PCT_cmax5_diff',
     'TS_PCT_season_cmax5_diff': 'TS_PCT_cmax5_diff',
     'USG_PCT_season_cmax5_diff': 'USG_PCT_cmax5_diff',
     'E_USG_PCT_season_cmax5_diff': 'E_USG_PCT_cmax5_diff',
     'E_PACE_season_cmax5_diff': 'E_PACE_cmax5_diff',
     'PACE_season_cmax5_diff': 'PACE_cmax5_diff',
     'PACE_PER40_season_cmax5_diff': 'PACE_PER40_cmax5_diff',
     'POSS_season_cmax5_diff': 'POSS_cmax5_diff',
     'PIE_season_cmax5_diff': 'PIE_cmax5_diff',
     'duration_minutes_season_cmax5_diff': 'duration_minutes_cmax5_diff',
     'w_pct_season_cmax5_diff': 'w_pct_cmax5_diff',
     'min_season_cmax5_diff': 'min_cmax5_diff',
     'fgm_season_cmax5_diff': 'fgm_cmax5_diff',
     'fga_season_cmax5_diff': 'fga_cmax5_diff',
     'fg_pct_season_cmax5_diff': 'fg_pct_cmax5_diff',
     'fg3m_season_cmax5_diff': 'fg3m_cmax5_diff',
     'fg3a_season_cmax5_diff': 'fg3a_cmax5_diff',
     'fg3_pct_season_cmax5_diff': 'fg3_pct_cmax5_diff',
     'ftm_season_cmax5_diff': 'ftm_cmax5_diff',
     'fta_season_cmax5_diff': 'fta_cmax5_diff',
     'ft_pct_season_cmax5_diff': 'ft_pct_cmax5_diff',
     'oreb_season_cmax5_diff': 'oreb_cmax5_diff',
     'dreb_season_cmax5_diff': 'dreb_cmax5_diff',
     'reb_season_cmax5_diff': 'reb_cmax5_diff',
     'ast_season_cmax5_diff': 'ast_cmax5_diff',
     'stl_season_cmax5_diff': 'stl_cmax5_diff',
     'blk_season_cmax5_diff': 'blk_cmax5_diff',
     'tov_season_cmax5_diff': 'tov_cmax5_diff',
     'pf_season_cmax5_diff': 'pf_cmax5_diff',
     'pts_season_cmax5_diff': 'pts_cmax5_diff',
     'score_difference_season_cmax5_diff': 'spread_cmax5_diff',
     'away_home_season_cmax5_diff': 'away_home__cmax5_diff',
     'E_OFF_RATING_season_cmax12_diff': 'E_OFF_RATING_cmax12_diff',
     'OFF_RATING_season_cmax12_diff': 'OFF_RATING_cmax12_diff',
     'E_DEF_RATING_season_cmax12_diff': 'E_DEF_RATING_cmax12_diff',
     'DEF_RATING_season_cmax12_diff': 'DEF_RATING_cmax12_diff',
     'E_NET_RATING_season_cmax12_diff': 'E_NET_RATING_cmax12_diff',
     'NET_RATING_season_cmax12_diff': 'NET_RATING_cmax12_diff',
     'AST_PCT_season_cmax12_diff': 'AST_PCT_cmax12_diff',
     'AST_TOV_season_cmax12_diff': 'AST_TOV_cmax12_diff',
     'AST_RATIO_season_cmax12_diff': 'AST_RATIO_cmax12_diff',
     'OREB_PCT_season_cmax12_diff': 'OREB_PCT_cmax12_diff',
     'DREB_PCT_season_cmax12_diff': 'DREB_PCT_cmax12_diff',
     'REB_PCT_season_cmax12_diff': 'REB_PCT_cmax12_diff',
     'E_TM_TOV_PCT_season_cmax12_diff': 'E_TM_TOV_PCT_cmax12_diff',
     'TM_TOV_PCT_season_cmax12_diff': 'TM_TOV_PCT_cmax12_diff',
     'EFG_PCT_season_cmax12_diff': 'EFG_PCT_cmax12_diff',
     'TS_PCT_season_cmax12_diff': 'TS_PCT_cmax12_diff',
     'USG_PCT_season_cmax12_diff': 'USG_PCT_cmax12_diff',
     'E_USG_PCT_season_cmax12_diff': 'E_USG_PCT_cmax12_diff',
     'E_PACE_season_cmax12_diff': 'E_PACE_cmax12_diff',
     'PACE_season_cmax12_diff': 'PACE_cmax12_diff',
     'PACE_PER40_season_cmax12_diff': 'PACE_PER40_cmax12_diff',
     'POSS_season_cmax12_diff': 'POSS_cmax12_diff',
     'PIE_season_cmax12_diff': 'PIE_cmax12_diff',
     'duration_minutes_season_cmax12_diff': 'duration_minutes_cmax12_diff',
     'w_pct_season_cmax12_diff': 'w_pct_cmax12_diff',
     'min_season_cmax12_diff': 'min_cmax12_diff',
     'fgm_season_cmax12_diff': 'fgm_cmax12_diff',
     'fga_season_cmax12_diff': 'fga_cmax12_diff',
     'fg_pct_season_cmax12_diff': 'fg_pct_cmax12_diff',
     'fg3m_season_cmax12_diff': 'fg3m_cmax12_diff',
     'fg3a_season_cmax12_diff': 'fg3a_cmax12_diff',
     'fg3_pct_season_cmax12_diff': 'fg3_pct_cmax12_diff',
     'ftm_season_cmax12_diff': 'ftm_cmax12_diff',
     'fta_season_cmax12_diff': 'fta_cmax12_diff',
     'ft_pct_season_cmax12_diff': 'ft_pct_cmax12_diff',
     'oreb_season_cmax12_diff': 'oreb_cmax12_diff',
     'dreb_season_cmax12_diff': 'dreb_cmax12_diff',
     'reb_season_cmax12_diff': 'reb_cmax12_diff',
     'ast_season_cmax12_diff': 'ast_cmax12_diff',
     'stl_season_cmax12_diff': 'stl_cmax12_diff',
     'blk_season_cmax12_diff': 'blk_cmax12_diff',
     'tov_season_cmax12_diff': 'tov_cmax12_diff',
     'pf_season_cmax12_diff': 'pf_cmax12_diff',
     'pts_season_cmax12_diff': 'pts_cmax12_diff',
     'score_difference_season_cmax12_diff': 'spread_cmax12_diff',
     'away_home_season_cmax12_diff': 'away_home__cmax12_diff',
     'E_OFF_RATING_season_cmax999_diff': 'E_OFF_RATING_cmax999_diff',
     'OFF_RATING_season_cmax999_diff': 'OFF_RATING_cmax999_diff',
     'E_DEF_RATING_season_cmax999_diff': 'E_DEF_RATING_cmax999_diff',
     'DEF_RATING_season_cmax999_diff': 'DEF_RATING_cmax999_diff',
     'E_NET_RATING_season_cmax999_diff': 'E_NET_RATING_cmax999_diff',
     'NET_RATING_season_cmax999_diff': 'NET_RATING_cmax999_diff',
     'AST_PCT_season_cmax999_diff': 'AST_PCT_cmax999_diff',
     'AST_TOV_season_cmax999_diff': 'AST_TOV_cmax999_diff',
     'AST_RATIO_season_cmax999_diff': 'AST_RATIO_cmax999_diff',
     'OREB_PCT_season_cmax999_diff': 'OREB_PCT_cmax999_diff',
     'DREB_PCT_season_cmax999_diff': 'DREB_PCT_cmax999_diff',
     'REB_PCT_season_cmax999_diff': 'REB_PCT_cmax999_diff',
     'E_TM_TOV_PCT_season_cmax999_diff': 'E_TM_TOV_PCT_cmax999_diff',
     'TM_TOV_PCT_season_cmax999_diff': 'TM_TOV_PCT_cmax999_diff',
     'EFG_PCT_season_cmax999_diff': 'EFG_PCT_cmax999_diff',
     'TS_PCT_season_cmax999_diff': 'TS_PCT_cmax999_diff',
     'USG_PCT_season_cmax999_diff': 'USG_PCT_cmax999_diff',
     'E_USG_PCT_season_cmax999_diff': 'E_USG_PCT_cmax999_diff',
     'E_PACE_season_cmax999_diff': 'E_PACE_cmax999_diff',
     'PACE_season_cmax999_diff': 'PACE_cmax999_diff',
     'PACE_PER40_season_cmax999_diff': 'PACE_PER40_cmax999_diff',
     'POSS_season_cmax999_diff': 'POSS_cmax999_diff',
     'PIE_season_cmax999_diff': 'PIE_cmax999_diff',
     'duration_minutes_season_cmax999_diff': 'duration_minutes_cmax999_diff',
     'w_pct_season_cmax999_diff': 'w_pct_cmax999_diff',
     'min_season_cmax999_diff': 'min_cmax999_diff',
     'fgm_season_cmax999_diff': 'fgm_cmax999_diff',
     'fga_season_cmax999_diff': 'fga_cmax999_diff',
     'fg_pct_season_cmax999_diff': 'fg_pct_cmax999_diff',
     'fg3m_season_cmax999_diff': 'fg3m_cmax999_diff',
     'fg3a_season_cmax999_diff': 'fg3a_cmax999_diff',
     'fg3_pct_season_cmax999_diff': 'fg3_pct_cmax999_diff',
     'ftm_season_cmax999_diff': 'ftm_cmax999_diff',
     'fta_season_cmax999_diff': 'fta_cmax999_diff',
     'ft_pct_season_cmax999_diff': 'ft_pct_cmax999_diff',
     'oreb_season_cmax999_diff': 'oreb_cmax999_diff',
     'dreb_season_cmax999_diff': 'dreb_cmax999_diff',
     'reb_season_cmax999_diff': 'reb_cmax999_diff',
     'ast_season_cmax999_diff': 'ast_cmax999_diff',
     'stl_season_cmax999_diff': 'stl_cmax999_diff',
     'blk_season_cmax999_diff': 'blk_cmax999_diff',
     'tov_season_cmax999_diff': 'tov_cmax999_diff',
     'pf_season_cmax999_diff': 'pf_cmax999_diff',
     'pts_season_cmax999_diff': 'pts_cmax999_diff',
     'score_difference_season_cmax999_diff': 'spread_cmax999_diff',
     'away_home_season_cmax999_diff': 'away_home__cmax999_diff'}

    column_name_dict_collection['matchup'] = \
    {'E_OFF_RATING_season_matchup_cma12_diff': 'E_OFF_RATING_sm_cma12_diff',
     'OFF_RATING_season_matchup_cma12_diff': 'OFF_RATING_sm_cma12_diff',
     'E_DEF_RATING_season_matchup_cma12_diff': 'E_DEF_RATING_sm_cma12_diff',
     'DEF_RATING_season_matchup_cma12_diff': 'DEF_RATING_sm_cma12_diff',
     'E_NET_RATING_season_matchup_cma12_diff': 'E_NET_RATING_sm_cma12_diff',
     'NET_RATING_season_matchup_cma12_diff': 'NET_RATING_sm_cma12_diff',
     'AST_PCT_season_matchup_cma12_diff': 'AST_PCT_sm_cma12_diff',
     'AST_TOV_season_matchup_cma12_diff': 'AST_TOV_sm_cma12_diff',
     'AST_RATIO_season_matchup_cma12_diff': 'AST_RATIO_sm_cma12_diff',
     'OREB_PCT_season_matchup_cma12_diff': 'OREB_PCT_sm_cma12_diff',
     'DREB_PCT_season_matchup_cma12_diff': 'DREB_PCT_sm_cma12_diff',
     'REB_PCT_season_matchup_cma12_diff': 'REB_PCT_sm_cma12_diff',
     'E_TM_TOV_PCT_season_matchup_cma12_diff': 'E_TM_TOV_PCT_sm_cma12_diff',
     'TM_TOV_PCT_season_matchup_cma12_diff': 'TM_TOV_PCT_sm_cma12_diff',
     'EFG_PCT_season_matchup_cma12_diff': 'EFG_PCT_sm_cma12_diff',
     'TS_PCT_season_matchup_cma12_diff': 'TS_PCT_sm_cma12_diff',
     'USG_PCT_season_matchup_cma12_diff': 'USG_PCT_sm_cma12_diff',
     'E_USG_PCT_season_matchup_cma12_diff': 'E_USG_PCT_sm_cma12_diff',
     'E_PACE_season_matchup_cma12_diff': 'E_PACE_sm_cma12_diff',
     'PACE_season_matchup_cma12_diff': 'PACE_sm_cma12_diff',
     'PACE_PER40_season_matchup_cma12_diff': 'PACE_PER40_sm_cma12_diff',
     'POSS_season_matchup_cma12_diff': 'POSS_sm_cma12_diff',
     'PIE_season_matchup_cma12_diff': 'PIE_sm_cma12_diff',
     'duration_minutes_season_matchup_cma12_diff': 'duration_minutes_sm_cma12_diff',
     'w_pct_season_matchup_cma12_diff': 'w_pct_sm_cma12_diff',
     'min_season_matchup_cma12_diff': 'min_sm_cma12_diff',
     'fgm_season_matchup_cma12_diff': 'fgm_sm_cma12_diff',
     'fga_season_matchup_cma12_diff': 'fga_sm_cma12_diff',
     'fg_pct_season_matchup_cma12_diff': 'fg_pct_sm_cma12_diff',
     'fg3m_season_matchup_cma12_diff': 'fg3m_sm_cma12_diff',
     'fg3a_season_matchup_cma12_diff': 'fg3a_sm_cma12_diff',
     'fg3_pct_season_matchup_cma12_diff': 'fg3_pct_sm_cma12_diff',
     'ftm_season_matchup_cma12_diff': 'ftm_sm_cma12_diff',
     'fta_season_matchup_cma12_diff': 'fta_sm_cma12_diff',
     'ft_pct_season_matchup_cma12_diff': 'ft_pct_sm_cma12_diff',
     'oreb_season_matchup_cma12_diff': 'oreb_sm_cma12_diff',
     'dreb_season_matchup_cma12_diff': 'dreb_sm_cma12_diff',
     'reb_season_matchup_cma12_diff': 'reb_sm_cma12_diff',
     'ast_season_matchup_cma12_diff': 'ast_sm_cma12_diff',
     'stl_season_matchup_cma12_diff': 'stl_sm_cma12_diff',
     'blk_season_matchup_cma12_diff': 'blk_sm_cma12_diff',
     'tov_season_matchup_cma12_diff': 'tov_sm_cma12_diff',
     'pf_season_matchup_cma12_diff': 'pf_sm_cma12_diff',
     'pts_season_matchup_cma12_diff': 'pts_sm_cma12_diff',
     'score_difference_season_matchup_cma12_diff': 'spread_sm_cma12_diff',
     'away_home_season_matchup_cma12_diff': 'away_home_sm_cma12_diff',
     'E_OFF_RATING_season_matchup_cmax12_diff': 'E_OFF_RATING_sm_cmax12_diff',
     'OFF_RATING_season_matchup_cmax12_diff': 'OFF_RATING_sm_cmax12_diff',
     'E_DEF_RATING_season_matchup_cmax12_diff': 'E_DEF_RATING_sm_cmax12_diff',
     'DEF_RATING_season_matchup_cmax12_diff': 'DEF_RATING_sm_cmax12_diff',
     'E_NET_RATING_season_matchup_cmax12_diff': 'E_NET_RATING_sm_cmax12_diff',
     'NET_RATING_season_matchup_cmax12_diff': 'NET_RATING_sm_cmax12_diff',
     'AST_PCT_season_matchup_cmax12_diff': 'AST_PCT_sm_cmax12_diff',
     'AST_TOV_season_matchup_cmax12_diff': 'AST_TOV_sm_cmax12_diff',
     'AST_RATIO_season_matchup_cmax12_diff': 'AST_RATIO_sm_cmax12_diff',
     'OREB_PCT_season_matchup_cmax12_diff': 'OREB_PCT_sm_cmax12_diff',
     'DREB_PCT_season_matchup_cmax12_diff': 'DREB_PCT_sm_cmax12_diff',
     'REB_PCT_season_matchup_cmax12_diff': 'REB_PCT_sm_cmax12_diff',
     'E_TM_TOV_PCT_season_matchup_cmax12_diff': 'E_TM_TOV_PCT_sm_cmax12_diff',
     'TM_TOV_PCT_season_matchup_cmax12_diff': 'TM_TOV_PCT_sm_cmax12_diff',
     'EFG_PCT_season_matchup_cmax12_diff': 'EFG_PCT_sm_cmax12_diff',
     'TS_PCT_season_matchup_cmax12_diff': 'TS_PCT_sm_cmax12_diff',
     'USG_PCT_season_matchup_cmax12_diff': 'USG_PCT_sm_cmax12_diff',
     'E_USG_PCT_season_matchup_cmax12_diff': 'E_USG_PCT_sm_cmax12_diff',
     'E_PACE_season_matchup_cmax12_diff': 'E_PACE_sm_cmax12_diff',
     'PACE_season_matchup_cmax12_diff': 'PACE_sm_cmax12_diff',
     'PACE_PER40_season_matchup_cmax12_diff': 'PACE_PER40_sm_cmax12_diff',
     'POSS_season_matchup_cmax12_diff': 'POSS_sm_cmax12_diff',
     'PIE_season_matchup_cmax12_diff': 'PIE_sm_cmax12_diff',
     'duration_minutes_season_matchup_cmax12_diff': 'duration_minutes_sm_cmax12_diff',
     'w_pct_season_matchup_cmax12_diff': 'w_pct_sm_cmax12_diff',
     'min_season_matchup_cmax12_diff': 'min_sm_cmax12_diff',
     'fgm_season_matchup_cmax12_diff': 'fgm_sm_cmax12_diff',
     'fga_season_matchup_cmax12_diff': 'fga_sm_cmax12_diff',
     'fg_pct_season_matchup_cmax12_diff': 'fg_pct_sm_cmax12_diff',
     'fg3m_season_matchup_cmax12_diff': 'fg3m_sm_cmax12_diff',
     'fg3a_season_matchup_cmax12_diff': 'fg3a_sm_cmax12_diff',
     'fg3_pct_season_matchup_cmax12_diff': 'fg3_pct_sm_cmax12_diff',
     'ftm_season_matchup_cmax12_diff': 'ftm_sm_cmax12_diff',
     'fta_season_matchup_cmax12_diff': 'fta_sm_cmax12_diff',
     'ft_pct_season_matchup_cmax12_diff': 'ft_pct_sm_cmax12_diff',
     'oreb_season_matchup_cmax12_diff': 'oreb_sm_cmax12_diff',
     'dreb_season_matchup_cmax12_diff': 'dreb_sm_cmax12_diff',
     'reb_season_matchup_cmax12_diff': 'reb_sm_cmax12_diff',
     'ast_season_matchup_cmax12_diff': 'ast_sm_cmax12_diff',
     'stl_season_matchup_cmax12_diff': 'stl_sm_cmax12_diff',
     'blk_season_matchup_cmax12_diff': 'blk_sm_cmax12_diff',
     'tov_season_matchup_cmax12_diff': 'tov_sm_cmax12_diff',
     'pf_season_matchup_cmax12_diff': 'pf_sm_cmax12_diff',
     'pts_season_matchup_cmax12_diff': 'pts_sm_cmax12_diff',
     'score_difference_season_matchup_cmax12_diff': 'spread_sm_cmax12_diff',
     'away_home_season_matchup_cmax12_diff': 'away_home_sm_cmax12_diff',
     'E_OFF_RATING_season_cma12_season_matchup_cmax12_diff': 'E_OFF_RATING_cma_sm_cmax12_diff',
     'OFF_RATING_season_cma12_season_matchup_cmax12_diff': 'OFF_RATING_cma_sm_cmax12_diff',
     'E_DEF_RATING_season_cma12_season_matchup_cmax12_diff': 'E_DEF_RATING_cma_sm_cmax12_diff',
     'DEF_RATING_season_cma12_season_matchup_cmax12_diff': 'DEF_RATING_cma_sm_cmax12_diff',
     'E_NET_RATING_season_cma12_season_matchup_cmax12_diff': 'E_NET_RATING_cma_sm_cmax12_diff',
     'NET_RATING_season_cma12_season_matchup_cmax12_diff': 'NET_RATING_cma_sm_cmax12_diff',
     'AST_PCT_season_cma12_season_matchup_cmax12_diff': 'AST_PCT_cma_sm_cmax12_diff',
     'AST_TOV_season_cma12_season_matchup_cmax12_diff': 'AST_TOV_cma_sm_cmax12_diff',
     'AST_RATIO_season_cma12_season_matchup_cmax12_diff': 'AST_RATIO_cma_sm_cmax12_diff',
     'OREB_PCT_season_cma12_season_matchup_cmax12_diff': 'OREB_PCT_cma_sm_cmax12_diff',
     'DREB_PCT_season_cma12_season_matchup_cmax12_diff': 'DREB_PCT_cma_sm_cmax12_diff',
     'REB_PCT_season_cma12_season_matchup_cmax12_diff': 'REB_PCT_cma_sm_cmax12_diff',
     'E_TM_TOV_PCT_season_cma12_season_matchup_cmax12_diff': 'E_TM_TOV_PCT_cma_sm_cmax12_diff',
     'TM_TOV_PCT_season_cma12_season_matchup_cmax12_diff': 'TM_TOV_PCT_cma_sm_cmax12_diff',
     'EFG_PCT_season_cma12_season_matchup_cmax12_diff': 'EFG_PCT_cma_sm_cmax12_diff',
     'TS_PCT_season_cma12_season_matchup_cmax12_diff': 'TS_PCT_cma_sm_cmax12_diff',
     'USG_PCT_season_cma12_season_matchup_cmax12_diff': 'USG_PCT_cma_sm_cmax12_diff',
     'E_USG_PCT_season_cma12_season_matchup_cmax12_diff': 'E_USG_PCT_cma_sm_cmax12_diff',
     'E_PACE_season_cma12_season_matchup_cmax12_diff': 'E_PACE_cma_sm_cmax12_diff',
     'PACE_season_cma12_season_matchup_cmax12_diff': 'PACE_cma_sm_cmax12_diff',
     'PACE_PER40_season_cma12_season_matchup_cmax12_diff': 'PACE_PER40_cma_sm_cmax12_diff',
     'POSS_season_cma12_season_matchup_cmax12_diff': 'POSS_cma_sm_cmax12_diff',
     'PIE_season_cma12_season_matchup_cmax12_diff': 'PIE_cma_sm_cmax12_diff',
     'duration_minutes_season_cma12_season_matchup_cmax12_diff': 'duration_minutes_cma_sm_cmax12_diff',
     'w_pct_season_cma12_season_matchup_cmax12_diff': 'w_pct_cma_sm_cmax12_diff',
     'min_season_cma12_season_matchup_cmax12_diff': 'min_cma_sm_cmax12_diff',
     'fgm_season_cma12_season_matchup_cmax12_diff': 'fgm_cma_sm_cmax12_diff',
     'fga_season_cma12_season_matchup_cmax12_diff': 'fga_cma_sm_cmax12_diff',
     'fg_pct_season_cma12_season_matchup_cmax12_diff': 'fg_pct_cma_sm_cmax12_diff',
     'fg3m_season_cma12_season_matchup_cmax12_diff': 'fg3m_cma_sm_cmax12_diff',
     'fg3a_season_cma12_season_matchup_cmax12_diff': 'fg3a_cma_sm_cmax12_diff',
     'fg3_pct_season_cma12_season_matchup_cmax12_diff': 'fg3_pct_cma_sm_cmax12_diff',
     'ftm_season_cma12_season_matchup_cmax12_diff': 'ftm_cma_sm_cmax12_diff',
     'fta_season_cma12_season_matchup_cmax12_diff': 'fta_cma_sm_cmax12_diff',
     'ft_pct_season_cma12_season_matchup_cmax12_diff': 'ft_pct_cma_sm_cmax12_diff',
     'oreb_season_cma12_season_matchup_cmax12_diff': 'oreb_cma_sm_cmax12_diff',
     'dreb_season_cma12_season_matchup_cmax12_diff': 'dreb_cma_sm_cmax12_diff',
     'reb_season_cma12_season_matchup_cmax12_diff': 'reb_cma_sm_cmax12_diff',
     'ast_season_cma12_season_matchup_cmax12_diff': 'ast_cma_sm_cmax12_diff',
     'stl_season_cma12_season_matchup_cmax12_diff': 'stl_cma_sm_cmax12_diff',
     'blk_season_cma12_season_matchup_cmax12_diff': 'blk_cma_sm_cmax12_diff',
     'tov_season_cma12_season_matchup_cmax12_diff': 'tov_cma_sm_cmax12_diff',
     'pf_season_cma12_season_matchup_cmax12_diff': 'pf_cma_sm_cmax12_diff',
     'pts_season_cma12_season_matchup_cmax12_diff': 'pts_cma_sm_cmax12_diff',
     'score_difference_season_cma12_season_matchup_cmax12_diff': 'spread_cma_sm_cmax12_diff',
     'away_home_season_cma12_season_matchup_cmax12_diff': 'away_home_cma_sm_cmax12_diff'}
    
    column_name_dict_collection['other5'] = \
    {'away_home_a': 'away_home_a',
     'losses_diff': 'losses_diff',
     'wins_diff': 'wins_diff',
     'strength_of_schedule_diff': 'strength_of_schedule_diff',
     'season_year': 'season_year'}
    
    return column_name_dict_collection


def get_column_name_dictionary_161():
    column_name_dict = {}
    
    column_name_dict_collection = \
    get_column_name_dict_collection()

    [column_name_dict.update(column_name_dict_collection[k]) for k in column_name_dict_collection.keys()]
    
    return column_name_dict





#################################################################################################################################
def preprocess_data_frame_for_numerics_161(df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference):

    random_state_2909 = 2909

    #sort strength of schedule diff, stat lost contribution diff, stat diff, stat matchup difference data frame
    df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference = \
    df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference.sort_values(['game_date', 'game_id'])



    #get categorical variable list
    column_name_list_categorical = \
    df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference.select_dtypes('object').columns.to_list() + ['team_id_a', 'team_id_b']

    #get indicator variables
    df_indicator_variables = \
    pd.get_dummies(df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference.loc[:, column_name_list_categorical])


    #get feature data frame
    df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference_not_categorical = \
    df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference.drop(columns=column_name_list_categorical)

    df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference_not_categorical = \
    df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference_not_categorical.drop(columns=['game_id', 'game_date'])

    df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference_not_categorical_features = \
    df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference_not_categorical.drop(columns=['score_difference_a'])


    #scale features data frame

    from sklearn.preprocessing import StandardScaler

    #create standard scaler object
    standard_scaler = StandardScaler() 


    # scale features with standard scaler object
    scaled_df_features_ndarray = standard_scaler.fit_transform(df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference_not_categorical_features) 
    

    #convert scaled features to data frame from type ndarray
    df_strength_of_schedule_diff_scaled_stat_lost_contribution_diff_scaled_stat_diff_scaled_stat_matchup_difference_scaled_not_categorical_features = \
    pd.DataFrame(scaled_df_features_ndarray, 
                 columns=df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference_not_categorical_features.columns)






    ##rename and reorder feature data frame for Machine Learning##

    #get colum name dictionary
    column_name_dict = \
    get_column_name_dictionary_161()

    column_name_dict_values = \
    [column_name_dict[k] for k in column_name_dict.keys()]


    #get feature dictionary values 
    column_name_dict_value_list_not_spread_a = \
    [column_name_dict[k] for k in column_name_dict.keys() if not 'spread_a' == column_name_dict[k]]

    #get dictionary keys
    column_name_dict_key_list_not_score_difference_a = \
    [k for k in column_name_dict.keys() if not 'score_difference_a' == k]


    #select columns for 161 model (Random Forest Regressor)
    df_strength_of_schedule_diff_scaled_stat_lost_contribution_diff_scaled_stat_diff_scaled_stat_matchup_difference_scaled_not_categorical_features_model_renamed = \
    df_strength_of_schedule_diff_scaled_stat_lost_contribution_diff_scaled_stat_diff_scaled_stat_matchup_difference_scaled_not_categorical_features\
    .loc[:, column_name_dict_key_list_not_score_difference_a].rename(columns=column_name_dict)


    #get column names of data frame fed into model in correct order
    df_161_train_split_time_series = \
    rcp('161_train_split_time_series.csv').drop(columns=['Unnamed: 0'])

    column_name_list_161_train_split_time_series = \
    df_161_train_split_time_series.columns.to_list()


    #reorder columns for Random Forest Regressor
    df_strength_of_schedule_diff_scaled_stat_lost_contribution_diff_scaled_stat_diff_scaled_stat_matchup_difference_scaled_not_categorical_features_model_renamed = \
    df_strength_of_schedule_diff_scaled_stat_lost_contribution_diff_scaled_stat_diff_scaled_stat_matchup_difference_scaled_not_categorical_features_model_renamed\
    .loc[:, df_161_train_split_time_series.drop(columns='spread_a').columns]






    ##get and rename target data frame##
    df_score_difference_a_target = \
    df_strength_of_schedule_diff_stat_lost_contribution_diff_stat_diff_stat_matchup_difference.loc[:, ['score_difference_a']]

    df_score_difference_a_target_renamed = \
    df_score_difference_a_target.rename(columns={'score_difference_a':'spread_a'})



    from sklearn.model_selection import train_test_split

    #split scaled features and target into train and test
    X_train, X_test, y_train, y_test = train_test_split(df_strength_of_schedule_diff_scaled_stat_lost_contribution_diff_scaled_stat_diff_scaled_stat_matchup_difference_scaled_not_categorical_features_model_renamed, 
                                                        df_score_difference_a_target_renamed, 
                                                        test_size=0.1, 
                                                        random_state=random_state_2909,
                                                        shuffle=False)

    #combine splits and store in data frame collection train and test
    data_frame_collection_train_test_161 = {}

    data_frame_collection_train_test_161['train'] = \
    pd.concat([X_train, y_train], axis=1)

    data_frame_collection_train_test_161['test'] = \
    pd.concat([X_test, y_test], axis=1)
    
    return data_frame_collection_train_test_161

#################################################################################################################################





#################################################################################################################################
def get_random_forest_time_series_cross_validation_index_8_train_test():
    #get train feature and target data frames

    #get train data frame
    df_train = rpp('161data_frame_collection_train_test.pkl',
                       parse_dates=False)['train']
    df_test = rpp('161data_frame_collection_train_test.pkl',
                      parse_dates=False)['test']

    df_train_test = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    df_time_series_cross_validation_index_8_train, df_time_series_cross_validation_index_8_test = \
    get_time_series_cross_validation_9_splits_train_test_index_8_data_frame(df_train_test)



    #select feature data frame X and target data frame Y

    #select feature data frame X
    X_time_series_cross_validation_index_8_train = \
    df_time_series_cross_validation_index_8_train.drop(columns = 'spread_a')

    #select target data frame
    Y_time_series_cross_validation_index_8_train = \
    df_time_series_cross_validation_index_8_train.loc[:, ['spread_a']]
    
    return X_time_series_cross_validation_index_8_train, Y_time_series_cross_validation_index_8_train, df_time_series_cross_validation_index_8_train, df_time_series_cross_validation_index_8_test
#################################################################################################################################











#################################################################################################################################
def get_column_name_dict_method_1_2():
    column_name_dict_method_1_2 = \
    {'game_id': 'game_id',
     'game_date': 'game_date',
     'team_id_a': 'team_id_a',
     'season': 'season',
     'team_id_b': 'team_id_b',
     'game_date_diff_diff': 'game_date_diff_diff',
     'Country/Region_a': 'Country/Region_a',
     'Country/Region_b': 'Country/Region_b',
     'travel_distance_diff': 'travel_distance_diff',
     'away_game_percent_day_window2_diff': 'away_game_percent_day_window2_diff',
     'home_game_percent_day_window2_diff': 'home_game_percent_day_window2_diff',
     'away_game_percent_day_window5_diff': 'away_game_percent_day_window5_diff',
     'home_game_percent_day_window5_diff': 'home_game_percent_day_window5_diff',
     'away_game_percent_day_window8_diff': 'away_game_percent_day_window8_diff',
     'home_game_percent_day_window8_diff': 'home_game_percent_day_window8_diff',
     'away_game_percent_day_window12_diff': 'away_game_percent_day_window12_diff',
     'home_game_percent_day_window12_diff': 'home_game_percent_day_window12_diff',
     'away_game_percent_day_window999_diff': 'away_game_percent_day_window999_diff',
     'home_game_percent_day_window999_diff': 'home_game_percent_day_window999_diff',
     'future_home_game_count_day_window2_diff': 'future_home_game_count_day_window2_diff',
     'future_game_count_day_window2_diff': 'future_game_count_day_window2_diff',
     'future_home_game_percent_day_window2_diff': 'future_home_game_percent_day_window2_diff',
     'future_home_game_count_day_window5_diff': 'future_home_game_count_day_window5_diff',
     'future_game_count_day_window5_diff': 'future_game_count_day_window5_diff',
     'future_home_game_percent_day_window5_diff': 'future_home_game_percent_day_window5_diff',
     'future_home_game_count_day_window8_diff': 'future_home_game_count_day_window8_diff',
     'future_game_count_day_window8_diff': 'future_game_count_day_window8_diff',
     'future_home_game_percent_day_window8_diff': 'future_home_game_percent_day_window8_diff',
     'future_home_game_count_day_window12_diff': 'future_home_game_count_day_window12_diff',
     'future_game_count_day_window12_diff': 'future_game_count_day_window12_diff',
     'future_home_game_percent_day_window12_diff': 'future_home_game_percent_day_window12_diff',
     'future_home_game_count_day_window999_diff': 'future_home_game_count_day_window999_diff',
     'future_game_count_day_window999_diff': 'future_game_count_day_window999_diff',
     'future_home_game_percent_day_window999_diff': 'future_home_game_percent_day_window999_diff',
     'game_latitude_diff': 'game_latitude_diff',
     'game_longitude_diff': 'game_longitude_diff',
     'future_away_game_count_day_window2_diff': 'future_Away_game_count_day_window2_diff',
     'future_away_game_percent_day_window2_diff': 'future_Away_game_percent_day_window2_diff',
     'future_away_game_count_day_window5_diff': 'future_Away_game_count_day_window5_diff',
     'future_away_game_percent_day_window5_diff': 'future_Away_game_percent_day_window5_diff',
     'future_away_game_count_day_window8_diff': 'future_Away_game_count_day_window8_diff',
     'future_away_game_percent_day_window8_diff': 'future_Away_game_percent_day_window8_diff',
     'future_away_game_count_day_window12_diff': 'future_Away_game_count_day_window12_diff',
     'future_away_game_percent_day_window12_diff': 'future_Away_game_percent_day_window12_diff',
     'future_away_game_count_day_window999_diff': 'future_Away_game_count_day_window999_diff',
     'future_away_game_percent_day_window999_diff': 'future_Away_game_percent_day_window999_diff',
     'game_date_diff_1_day_season_csum2_diff': 'games_back_to_back_count_game_window2_diff',
     'game_date_diff_1_day_season_csum5_diff': 'games_back_to_back_count_game_window5_diff',
     'game_date_diff_1_day_season_csum12_diff': 'games_back_to_back_count_game_window12_diff',
     'game_date_diff_1_day_season_csum999_diff': 'games_back_to_back_count_game_window999_diff',
     'game_date_diff_1_day_season_cma2_diff': 'games_back_to_back_percentage_game_window2_diff',
     'game_date_diff_1_day_season_cma5_diff': 'games_back_to_back_percentage_game_window5_diff',
     'game_date_diff_1_day_season_cma12_diff': 'games_back_to_back_percentage_game_window12_diff',
     'game_date_diff_1_day_season_cma999_diff': 'games_back_to_back_percentage_game_window999_diff',
     'away_game_day_window_csum2_diff': 'away_game_count_day_window2_diff',
     'home_game_day_window_csum2_diff': 'home_game_count_day_window2_diff',
     'game_day_window_csum2_diff': 'game_count_day_window2_diff',
     'away_game_day_window_csum5_diff': 'away_game_count_day_window5_diff',
     'home_game_day_window_csum5_diff': 'home_game_count_day_window5_diff',
     'game_day_window_csum5_diff': 'game_count_day_window5_diff',
     'away_game_day_window_csum8_diff': 'away_game_count_day_window8_diff',
     'home_game_day_window_csum8_diff': 'home_game_count_day_window8_diff',
     'game_day_window_csum8_diff': 'game_count_day_window8_diff',
     'away_game_day_window_csum12_diff': 'away_game_count_day_window12_diff',
     'home_game_day_window_csum12_diff': 'home_game_count_day_window12_diff',
     'game_day_window_csum12_diff': 'game_count_day_window12_diff',
     'away_game_day_window_csum999_diff': 'away_game_count_day_window999_diff',
     'home_game_day_window_csum999_diff': 'home_game_count_day_window999_diff',
     'game_day_window_csum999_diff': 'game_count_day_window999_diff',
     'away_home_diff': 'away_home__diff',
     'city_proper_metro_area_GDP_bil_diff': 'city_proper_metro_Area_GDP_bil_diff',
     'city_proper_metro_area_GDP_bil1_diff': 'city_proper_metro_Area_GDP_bil1_diff',
     'game_date_diff_1_day_diff': 'game_date_diff1_diff',
     'game_city_time_zone_a': 'game_time_zone_b',
     'game_city_time_zone_b': 'game_time_zone_a',
     'city_proper_metro_area_a': 'city_proper_metro_Area_a',
     'city_proper_metro_area_b': 'city_proper_metro_Area_b',
     'team_city_time_zone_a': 'time_zone_a',
     'team_city_time_zone_b': 'time_zone_b',
     'geographic_team_city_name_latitude_diff': 'latitude_diff',
     'geographic_team_city_name_longitude_diff': 'longitude_diff',
     'time_zone_diff_diff': 'time_zone_diff_a',
     'time_zone_diff_abs_diff': 'time_zone_diff_abs_a',
     'score_difference_a':'spread_a'}
    
    return column_name_dict_method_1_2
#################################################################################################################################



#################################################################################################################################
def get_172_preprocessed_by_column_name_dictionary_and_game_latitude_diff_game_longitude_diff_values(
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b,
    df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window,
    method_1_2=1):

    #get 172_train_split_time_series and 172_test_split_time_series


    # df_72_71C_schedule_location_differences = \
    # rcp('72_71C_schedule_location_differences_2010_2018.csv')




    #add game latitutde and game longitutde with fill with zero
    df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window\
    .loc[:, ['game_latitude_diff', 'game_longitude_diff']] = 0


    #get score_difference_a
    df_game_id_team_id_a_team_id_b_score_difference_a = \
    df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['a_b']\
    .loc[:, ['game_id', 'team_id_a', 'team_id_b', 'score_difference_a']]

    #add score_difference_a
    df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a = \
    merge_data_frame_list([df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window,
                               df_game_id_team_id_a_team_id_b_score_difference_a])


    #get column name dictionary for modeling
    column_name_dict_method_1_2 = \
    get_column_name_dict_method_1_2()

    #get column name list for column selection using dictionary keys
    column_name_list = \
    list(column_name_dict_method_1_2.keys())




    #select and rename columns for feeding into random forest regressor
    method_1_2 = 1
    if method_1_2 == 1:

        #fill method 1: fill with zero

        #create and set columns time_zone_diff_a and time_zone_diff_abs_a to 0
        df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a\
        .loc[:, ['time_zone_diff_diff', 'time_zone_diff_abs_diff']] = 0


        #select and rename columns by dictionary values
        df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed = \
        df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a\
        .loc[:, column_name_list].rename(columns=column_name_dict_method_1_2)


    elif method_1_2 == 2:

        #fill method 2: fill with approximation
        df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed = \
        df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a\
        .loc[:, column_name_list].rename(columns=column_name_dict_method_1_2)



    #convert team_id_a and team_id_b to type object
    df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed.loc[:, ['team_id_a', 'team_id_b']] = \
    df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed.loc[:, ['team_id_a', 'team_id_b']].astype('object')
    
    
    #get categorical column name list
    column_name_list_categorical = \
    list(df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed.select_dtypes('object').columns)


    #get column name list game_id and game_date
    column_name_list_game_id_game_date = ['game_id', 'game_date']



    #get indicator variable data frame
    df_indicator_variables = \
    pd.get_dummies(
        df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed\
        .loc[:, column_name_list_categorical])



    #drop categorical variable by categorical column name list
    df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed_numerics = \
    df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed\
    .drop(columns=column_name_list_categorical)

    #drop game id and game date columns
    df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed_numerics = \
    df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed_numerics\
    .drop(columns=column_name_list_game_id_game_date)


    #get numeric feature data frame
    df_features_numeric = \
    df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed_numerics\
    .drop(columns='spread_a')


    #get target variable data frame
    df_target = \
    df_city_proper_metro_area_gdp_latitude_longitude_travel_distance_away_home_game_percentage_day_window_future_away_home_game_percentage_day_window_score_difference_a_renamed_numerics.loc[:, ['spread_a']]

    
    return df_features_numeric, \
           df_target, \
           df_indicator_variables
#################################################################################################################################





#################################################################################################################################
def get_172_preprocessed_by_scaling_and_train_test_splits(df_features_numeric,
                                                          df_target,
                                                          df_indicator_variables):

    #scale features

    #create standard scaler object
    from sklearn.preprocessing import StandardScaler
    standard_scaler = StandardScaler() 

    #scale numeric features
    scaled_df_features_ndarray = \
    standard_scaler.fit_transform(df_features_numeric) 


    #convert numeric featurs back to data frame
    df_features_numeric_scaled = \
    pd.DataFrame(data=scaled_df_features_ndarray, 
                 columns=df_features_numeric.columns)



    #split scaled features and target into train and test
    from sklearn.model_selection import train_test_split
    random_state_2909 = 2909
    X_train, X_test, y_train, y_test, X_train_indicator_variables, X_test_indicator_variables = \
    train_test_split(df_features_numeric_scaled, 
                     df_target,
                     df_indicator_variables,
                     test_size=0.1, 
                     random_state=random_state_2909,
                     shuffle=False)

    #combine splits and store in data frame collection train and test
    data_frame_collection_train_test_172 = {}

    data_frame_collection_train_test_172['train'] = \
    pd.concat([X_train, y_train], axis=1)

    data_frame_collection_train_test_172['test'] = \
    pd.concat([X_test, y_test], axis=1)
    
    #combine splits for indicator variables in data frame collection train and test
    data_frame_collection_train_test_172_indicator_variables = {}
    
    data_frame_collection_train_test_172_indicator_variables['train'] = X_train_indicator_variables
    
    data_frame_collection_train_test_172_indicator_variables['test'] = X_test_indicator_variables
    
    
    return data_frame_collection_train_test_172, data_frame_collection_train_test_172_indicator_variables
#################################################################################################################################








'''
Modeling functions
'''


def return_saved_model_if_it_exists(filename):
    import pickle
    
    relative_directory_path = os.path.join('..', '05_models')

    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)

    relative_file_path = os.path.join(relative_directory_path, filename)

    if os.path.exists(relative_file_path):
        print('This file already exists')
        
        with (open(relative_file_path, "rb")) as openfile:
            model_readback = pickle.load(openfile)
        
        return model_readback
    
    else:
        return None


def save_and_return_model(model,
                          filename):

    import pickle

    relative_directory_path = os.path.join('..', '05_models')

    #make relative file direactory path if it doesn't exist
    if not os.path.exists(relative_directory_path):
        os.mkdir(relative_directory_path)
        
    #get relative file path name
    relative_file_path = os.path.join(relative_directory_path, filename)

    #if model file already exists, say it
    if os.path.exists(relative_file_path):
            print('This file already exists.')

    #if model file doesn't exist, then save it
    elif not os.path.exists(relative_file_path):
        file_object_wb =  open(relative_file_path, "wb")
        pickle.dump(model, file_object_wb)
        file_object_wb.close()
    
    #readback model file
    with (open(relative_file_path, "rb")) as open_file:
        model_readback = pickle.load(open_file)

    return model_readback



def get_random_forest_hyperparameter_tuning_results_data_frame():
    results_list = \
    [[5, 200, 5, 50, 0.20115401542812372], 
     [5, 200, 5, 100, 0.2008693419323584], 
     [5, 200, 10, 50, 0.19931527393198556], 
     [5, 200, 10, 100, 0.19676780047915843], 
     [5, 200, 20, 50, 0.19476471253366656], 
     [5, 200, 20, 100, 0.1946636324278841], 
     [6, 200, 5, 50, 0.20503784910623069], 
     [6, 200, 5, 100, 0.20479253774867656], 
     [6, 200, 10, 50, 0.2056633916310161], 
     [6, 200, 10, 100, 0.20651398197050308], 
     [6, 200, 20, 50, 0.20448605830461608], 
     [6, 200, 20, 100, 0.20767687869829599], 
     [7, 200, 5, 50, 0.13308888337103664], 
     [7, 200, 5, 100, 0.13363673914620122], 
     [7, 200, 10, 50, 0.12796014321359905], 
     [7, 200, 10, 100, 0.13189259562556166], 
     [7, 200, 20, 50, 0.12680487781455707], 
     [7, 200, 20, 100, 0.13412125292654864]]
    
    df_time_series_cross_validation_score = pd.DataFrame(results_list, 
                                                         columns=['index', 'n_estimators', 'max_depth', 'min_samples_split', 'score'])


    return df_time_series_cross_validation_score

def get_random_forest_hyperparameter_tuning_median_score():
    '''
    Time Series Cross Validation Median Score.'''
    
    df_time_series_cross_validation_score = \
    get_random_forest_hyperparameter_tuning_results_data_frame()

    df_time_series_cross_validation_score_median = \
    df_time_series_cross_validation_score.loc[:, ['n_estimators', 'max_depth', 'min_samples_split', 'score']]\
    .groupby(['n_estimators', 'max_depth', 'min_samples_split']).median().reset_index()\
    .sort_values(['score'], ascending=False).rename(columns={'score':'median_score'})

    return df_time_series_cross_validation_score_median








def get_best_random_forest_hyperparameters(data_frame_collection_train_test,
                                           run_hyperparameter_tuning_model_train_yes_no='no'):
    
    
    #get train features and target
    X_train = \
    data_frame_collection_train_test['train'].drop(columns = 'spread_a')

    y_train = \
    data_frame_collection_train_test['train'].loc[:, 'spread_a']
    
    
    
    if run_hyperparameter_tuning_model_train_yes_no == 'yes':

        #get test features and target
        X_test = \
        data_frame_collection_train_test['test'].drop(columns='spread_a')

        y_test = \
        data_frame_collection_train_test['test'].loc[:, 'spread_a']


        #build grid parameter dictionary
        grid_params = {
            'randomforestregressor__n_estimators': [200],
            'randomforestregressor__max_depth': [5, 10, 20],
            'randomforestregressor__min_samples_split': [50, 100]
        }


        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import RandomForestRegressor


        tscv = TimeSeriesSplit(n_splits=8)
        i = 0
        score = []
        for train_index, validation_index in tscv.split(X_train):

            if len(train_index) > 6159:
                print(train_index)
                print(validation_index)
                print()

                X_tr, X_val = X_train.iloc[train_index], X_train.iloc[validation_index]
                y_tr, y_val = y_train.iloc[train_index], y_train.iloc[validation_index]
                for number_estimators in grid_params['randomforestregressor__n_estimators']:
                    for max_depth in grid_params['randomforestregressor__max_depth']:
                        for min_samples_split in grid_params['randomforestregressor__min_samples_split']:
                            print(number_estimators)
                            print(max_depth)
                            print(min_samples_split)
                            rfr = RandomForestRegressor(n_estimators=int(number_estimators),
                                                        max_depth=int(max_depth),
                                                        min_samples_split=int(min_samples_split),
                                                        random_state=47,
                                                        criterion = 'mae',
                                                        n_jobs=-1)
                            rfr.fit(X_tr, y_tr)
                            score.append([i, 
                                          number_estimators,
                                          max_depth, 
                                          min_samples_split, 
                                          rfr.score(X_val, y_val)])
                            print(score)
                            print()
            else:
                print('skipped')
            i += 1
            
    elif run_hyperparameter_tuning_model_train_yes_no == 'no':
        print("if run_hyperparameter_tuning_model_train_yes_no='yes', then index structure would be...")
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=8)
        i = 1
        score = []
        for train_index, validation_index in tscv.split(X_train):

            if len(train_index) > 6159:

                print('train indices: ' + str(len(train_index)))
                print('validation indices: ' + str(len(validation_index)) + '\n')

            i += 1
    else:
        print('input argument error')

        
        
def get_time_series_cross_validation_9_splits_train_test_size_data_frame():

    from sklearn.model_selection import TimeSeriesSplit
    df_time_series_cross_validation_9_splits_index_count = pd.DataFrame()
    

    
    #get train test data frame
    df_train = rpp('161data_frame_collection_train_test.pkl',
                       parse_dates=False)['train']
    
    df_test = rpp('161data_frame_collection_train_test.pkl',
                      parse_dates=False)['test']
    
    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    
    
    #build time series cross validation train and test index count data frame
    time_series_cross_validation_9_splits = TimeSeriesSplit(max_train_size=None, n_splits=9)

    for i, (train_index_ndarray, test_index_ndarray) in enumerate(time_series_cross_validation_9_splits.split(df)):
        train_index_ndarray_len = len(train_index_ndarray)
        test_index_ndarray_len = len(test_index_ndarray)

        df_temp = \
        pd.DataFrame({'train_size':[train_index_ndarray_len], 
                      'test_size':[test_index_ndarray_len]})
        
        df_time_series_cross_validation_9_splits_index_count = pd.concat([df_time_series_cross_validation_9_splits_index_count, df_temp])

    return df_time_series_cross_validation_9_splits_index_count.reset_index(drop=True).reset_index().rename(columns={'index':'tscv9_index'})



def get_time_series_cross_validation_9_splits_train_test_index_8_data_frame(df):
    df_time_series_cross_validation_9_splits_train_test_size = \
    get_time_series_cross_validation_9_splits_train_test_size_data_frame()
    
    train_size = df_time_series_cross_validation_9_splits_train_test_size.loc[8, 'train_size']
    
    test_size = df_time_series_cross_validation_9_splits_train_test_size.loc[8, 'test_size']
    
    train_index_list = [k for k in range(train_size)]
    
    test_index_list = [train_size + k for k in range(test_size)]
    
    return df.iloc[train_index_list, :], df.iloc[test_index_list, :] 




def get_random_forest_regressor_time_series_cross_validation_results_collection(X, Y):
    '''get Time Series Cross Validation of the Train Data'''
    from sklearn.model_selection import cross_validate
    from sklearn.ensemble import RandomForestRegressor
    
    from sklearn.model_selection import TimeSeriesSplit

    time_series_cross_validation_8_splits = \
    TimeSeriesSplit(n_splits=8,
                    max_train_size=None)

    rf = RandomForestRegressor(n_estimators=200,
                               min_samples_split=50,
                               max_depth=5,
                               random_state=47,
                               criterion = 'mae')

    random_forest_time_series_cross_validation_8_splits_train_results_collection = cross_validate(rf, 
                                                                                                  X, 
                                                                                                  Y, 
                                                                                                  cv=time_series_cross_validation_8_splits.split(df), 
                                                                                                  return_estimator=True)

    return random_forest_time_series_cross_validation_8_splits_train_results_collection




#get Time Series Cross Validation 8 Splits Validation Feature Collection and Train Feature Collection

#get 161_time_series_cross_validation_8_splits_validation_score_difference_a_predictions_trial_2


#df = df_time_series_cross_validation_train_index_8

def get_df_time_series_cross_validation_8_splits_score_difference_a_predicted(
    df_time_series_cross_validation_index_8_train,
    X_time_series_cross_validation_index_8_train,
    random_forest_time_series_cross_validation_8_splits_train_results_collection):

    #variable initializations
    X_train_collection = {}
    X_validation_collection = {}

    tscv8_validation_predictions_collection = {}

    tscv8_validation_predictions_list = []




    #get Time Series Cross Validation 8 Splits Validation Feature Collection and Train Feature Collection
    from sklearn.model_selection import TimeSeriesSplit

    time_series_cross_validation_8_splits = TimeSeriesSplit(max_train_size=None, 
                                                            n_splits=8)

    for i, (train_index, validation_index) in enumerate(time_series_cross_validation_8_splits.split(df_time_series_cross_validation_index_8_train)):
        X_train_collection[i], X_validation_collection[i] = X_time_series_cross_validation_index_8_train.iloc[train_index, :], \
                                                            X_time_series_cross_validation_index_8_train.iloc[validation_index, :]


    #get Time Series Cross Validation 8 Splits 'spread_a' (i.e. Score Difference a) predictions
    for i, random_forest_regressor in enumerate(random_forest_time_series_cross_validation_8_splits_train_results_collection.get('estimator')):
        tscv8_validation_predictions_collection[i] = random_forest_regressor.predict(X_validation_collection[i])


    #convert Time Series Cross Validation 8 Splits 'spread_a' Predictions collection to data frame
    for i in range(8):
        tscv8_validation_predictions_list += list(tscv8_validation_predictions_collection[i])

    df_time_series_cross_validation_8_splits_score_difference_a_predicted = \
    pd.DataFrame({'score_difference_a_predicted': tscv8_validation_predictions_list})

    return df_time_series_cross_validation_8_splits_score_difference_a_predicted




def get_df_score_difference_a_actual(df_time_series_cross_validation_index_8_train):

    #get validation index list

    #initialize validation index list
    validation_index_list = []

    from sklearn.model_selection import TimeSeriesSplit
    time_series_cross_validation_8_splits = TimeSeriesSplit(max_train_size=None, 
                                                            n_splits=8)

    for i, (train_index_ndarray, validation_index_ndarray) in enumerate(time_series_cross_validation_8_splits.split(df_time_series_cross_validation_index_8_train)):
        validation_index_list += list(validation_index_ndarray)


    #select score difference actual by indices of Times Series Cross Validation 8 Splits Validation Rows
    df_score_difference_a_actual = df_time_series_cross_validation_index_8_train.loc[:, ['spread_a']]
    df_score_difference_a_actual = df_score_difference_a_actual.iloc[validation_index_list, :]
    
    return df_score_difference_a_actual





def get_particular_folds_of_10_folds(df,
                                     df_number_folds,
                                     get_fold_indices):

    from sklearn.model_selection import TimeSeriesSplit
    
    n_splits = df_number_folds - 1
    
    tscv = TimeSeriesSplit(n_splits = n_splits)
    
    df_collection = {}
    
    for i, (train_indices, test_indices) in enumerate(tscv.split(df)):

        if i + 2 in get_fold_indices:
            df_collection[i + 2] = df.iloc[test_indices, :]

    df_filtered = pd.DataFrame()
    for i in get_fold_indices:
        df_filtered = pd.concat([df_filtered, df_collection[i]])
    
    
    return df_filtered







def get_best_random_forest_model_172(X_train, y_train):
    from sklearn.pipeline import make_pipeline
    
    from sklearn.ensemble import RandomForestRegressor

    from sklearn.model_selection import cross_validate, GridSearchCV


    #hyperparameter tuning
    RF_pipe = make_pipeline( RandomForestRegressor(criterion = 'mae', random_state=47))


    grid_params = {'randomforestregressor__n_estimators': [200], 'randomforestregressor__max_depth': [5, 10, 20, 1000], 'randomforestregressor__min_samples_split': [50, 100, 1000]}


    rf_grid_cv = GridSearchCV(RF_pipe, param_grid=grid_params, cv=5, n_jobs=-1)

    rf_grid_cv.fit(X_train, y_train)

    rf_best_cv_results = cross_validate(rf_grid_cv.best_estimator_, X_train, y_train, cv=5)


    imps = rf_grid_cv.best_estimator_.named_steps.randomforestregressor.feature_importances_
    rf_feat_imps = pd.Series(imps, index=X_train.columns).sort_values(ascending=False)

    
    return rf_grid_cv.best_estimator_






def get_column_name_list_172_trial_2(list_1_2=1):
    
    if list_1_2 == 1:
        column_name_list_172_trial_2 =\
        ['travel_distance_diff',
         'latitude_diff',
         'longitude_diff',
         'games_back_to_back_percentage_game_window999_diff',
         'game_count_day_window999_diff',
         'away_game_count_day_window2_diff',
         'away_game_percent_day_window999_diff',
         'away_game_percent_day_window12_diff',
         'game_date_diff_diff',
         'home_game_percent_day_window999_diff',
         'future_Away_game_percent_day_window999_diff',
         'game_count_day_window12_diff',
         'future_home_game_percent_day_window999_diff',
         'games_back_to_back_count_game_window999_diff',
         'games_back_to_back_percentage_game_window12_diff',
         'future_home_game_percent_day_window12_diff',
         'away_game_count_day_window12_diff',
         'future_Away_game_percent_day_window12_diff',
         'game_count_day_window2_diff',
         'away_home__diff',
         'game_date_diff1_diff',
         'future_home_game_count_day_window12_diff',
         'home_game_percent_day_window12_diff',
         'future_Away_game_percent_day_window8_diff',
         'away_game_count_day_window999_diff',
         'future_Away_game_count_day_window5_diff',
         'home_game_percent_day_window8_diff',
         'future_Away_game_count_day_window8_diff',
         'home_game_count_day_window12_diff',
         'game_count_day_window5_diff',
         'future_game_count_day_window12_diff',
         'future_home_game_count_day_window8_diff',
         'games_back_to_back_count_game_window12_diff',
         'home_game_count_day_window8_diff',
         'future_home_game_percent_day_window8_diff',
         'games_back_to_back_percentage_game_window5_diff',
         'away_game_percent_day_window5_diff',
         'away_game_percent_day_window8_diff',
         'home_game_count_day_window999_diff',
         'home_game_count_day_window5_diff',
         'future_game_count_day_window2_diff',
         'away_game_percent_day_window2_diff',
         'future_game_count_day_window8_diff',
         'future_game_count_day_window999_diff',
         'future_home_game_percent_day_window5_diff',
         'future_Away_game_percent_day_window5_diff',
         'game_count_day_window8_diff',
         'future_game_count_day_window5_diff',
         'home_game_percent_day_window5_diff',
         'time_zone_diff_a',
         'future_Away_game_count_day_window12_diff',
         'future_Away_game_count_day_window999_diff',
         'future_home_game_count_day_window2_diff',
         'future_home_game_count_day_window999_diff',
         'future_home_game_count_day_window5_diff',
         'away_game_count_day_window8_diff',
         'games_back_to_back_count_game_window2_diff',
         'time_zone_diff_abs_a',
         'away_game_count_day_window5_diff',
         'home_game_count_day_window2_diff',
         'games_back_to_back_count_game_window5_diff',
         'home_game_percent_day_window2_diff',
         'games_back_to_back_percentage_game_window2_diff',
         'future_Away_game_count_day_window2_diff',
         'future_Away_game_percent_day_window2_diff',
         'future_home_game_percent_day_window2_diff',
         'game_longitude_diff',
         'game_latitude_diff']
    elif list_1_2 == 2:
        column_name_list_172_trial_2 = \
        ['time_zone_diff_a',
         'time_zone_diff_abs_a',
         'away_home__diff',
         'latitude_diff',
         'longitude_diff',
         'game_latitude_diff',
         'game_longitude_diff',
         'travel_distance_diff',
         'game_date_diff_diff',
         'game_date_diff1_diff',
         'games_back_to_back_count_game_window2_diff',
         'games_back_to_back_count_game_window5_diff',
         'games_back_to_back_count_game_window12_diff',
         'games_back_to_back_count_game_window999_diff',
         'games_back_to_back_percentage_game_window2_diff',
         'games_back_to_back_percentage_game_window5_diff',
         'games_back_to_back_percentage_game_window12_diff',
         'games_back_to_back_percentage_game_window999_diff',
         'away_game_count_day_window2_diff',
         'home_game_count_day_window2_diff',
         'game_count_day_window2_diff',
         'away_game_percent_day_window2_diff',
         'home_game_percent_day_window2_diff',
         'away_game_count_day_window5_diff',
         'home_game_count_day_window5_diff',
         'game_count_day_window5_diff',
         'away_game_percent_day_window5_diff',
         'home_game_percent_day_window5_diff',
         'away_game_count_day_window8_diff',
         'home_game_count_day_window8_diff',
         'game_count_day_window8_diff',
         'away_game_percent_day_window8_diff',
         'home_game_percent_day_window8_diff',
         'away_game_count_day_window12_diff',
         'home_game_count_day_window12_diff',
         'game_count_day_window12_diff',
         'away_game_percent_day_window12_diff',
         'home_game_percent_day_window12_diff',
         'away_game_count_day_window999_diff',
         'home_game_count_day_window999_diff',
         'game_count_day_window999_diff',
         'away_game_percent_day_window999_diff',
         'home_game_percent_day_window999_diff',
         'future_Away_game_count_day_window2_diff',
         'future_home_game_count_day_window2_diff',
         'future_game_count_day_window2_diff',
         'future_Away_game_percent_day_window2_diff',
         'future_home_game_percent_day_window2_diff',
         'future_Away_game_count_day_window5_diff',
         'future_home_game_count_day_window5_diff',
         'future_game_count_day_window5_diff',
         'future_Away_game_percent_day_window5_diff',
         'future_home_game_percent_day_window5_diff',
         'future_Away_game_count_day_window8_diff',
         'future_home_game_count_day_window8_diff',
         'future_game_count_day_window8_diff',
         'future_Away_game_percent_day_window8_diff',
         'future_home_game_percent_day_window8_diff',
         'future_Away_game_count_day_window12_diff',
         'future_home_game_count_day_window12_diff',
         'future_game_count_day_window12_diff',
         'future_Away_game_percent_day_window12_diff',
         'future_home_game_percent_day_window12_diff',
         'future_Away_game_count_day_window999_diff',
         'future_home_game_count_day_window999_diff',
         'future_game_count_day_window999_diff',
         'future_Away_game_percent_day_window999_diff',
         'future_home_game_percent_day_window999_diff']
        

    return column_name_list_172_trial_2




def get_column_name_list_172_trial_4_ordered():

    column_name_list_172_trial_4_ordered = \
    ['travel_distance_diff',
     'latitude_diff',
     'longitude_diff',
     'games_back_to_back_percentage_game_window999_diff',
     'game_count_day_window999_diff',
     'away_game_count_day_window2_diff',
     'away_game_percent_day_window999_diff',
     'away_game_percent_day_window12_diff',
     'game_date_diff_diff',
     'home_game_percent_day_window999_diff',
     'future_Away_game_percent_day_window999_diff',
     'game_count_day_window12_diff',
     'future_home_game_percent_day_window999_diff',
     'games_back_to_back_count_game_window999_diff',
     'games_back_to_back_percentage_game_window12_diff',
     'future_home_game_percent_day_window12_diff',
     'away_game_count_day_window12_diff',
     'future_Away_game_percent_day_window12_diff',
     'game_count_day_window2_diff',
     'away_home__diff',
     'game_date_diff1_diff',
     'future_home_game_count_day_window12_diff',
     'home_game_percent_day_window12_diff',
     'future_Away_game_percent_day_window8_diff',
     'away_game_count_day_window999_diff',
     'future_Away_game_count_day_window5_diff',
     'home_game_percent_day_window8_diff',
     'future_Away_game_count_day_window8_diff',
     'home_game_count_day_window12_diff',
     'game_count_day_window5_diff',
     'future_game_count_day_window12_diff',
     'future_home_game_count_day_window8_diff',
     'games_back_to_back_count_game_window12_diff',
     'home_game_count_day_window8_diff',
     'future_home_game_percent_day_window8_diff',
     'games_back_to_back_percentage_game_window5_diff',
     'away_game_percent_day_window5_diff',
     'away_game_percent_day_window8_diff',
     'home_game_count_day_window999_diff',
     'home_game_count_day_window5_diff',
     'future_game_count_day_window2_diff',
     'away_game_percent_day_window2_diff',
     'future_game_count_day_window8_diff',
     'future_game_count_day_window999_diff',
     'future_home_game_percent_day_window5_diff',
     'future_Away_game_percent_day_window5_diff',
     'game_count_day_window8_diff',
     'future_game_count_day_window5_diff',
     'home_game_percent_day_window5_diff',
     'time_zone_diff_a',
     'team_id_a_1610612737',
     'team_id_a_1610612738',
     'team_id_a_1610612739',
     'team_id_a_1610612740',
     'team_id_a_1610612741',
     'team_id_a_1610612742',
     'team_id_a_1610612743',
     'team_id_a_1610612744',
     'team_id_a_1610612745',
     'team_id_a_1610612746',
     'team_id_a_1610612747',
     'team_id_a_1610612748',
     'team_id_a_1610612749',
     'team_id_a_1610612750',
     'team_id_a_1610612751',
     'team_id_a_1610612752',
     'team_id_a_1610612753',
     'team_id_a_1610612754',
     'team_id_a_1610612755',
     'team_id_a_1610612756',
     'team_id_a_1610612757',
     'team_id_a_1610612758',
     'team_id_a_1610612759',
     'team_id_a_1610612760',
     'team_id_a_1610612761',
     'team_id_a_1610612762',
     'team_id_a_1610612763',
     'team_id_a_1610612765',
     'team_id_a_1610612766',
     'team_id_b_1610612738',
     'team_id_b_1610612739',
     'team_id_b_1610612740',
     'team_id_b_1610612741',
     'team_id_b_1610612742',
     'team_id_b_1610612743',
     'team_id_b_1610612744',
     'team_id_b_1610612745',
     'team_id_b_1610612746',
     'team_id_b_1610612747',
     'team_id_b_1610612748',
     'team_id_b_1610612749',
     'team_id_b_1610612750',
     'team_id_b_1610612751',
     'team_id_b_1610612752',
     'team_id_b_1610612753',
     'team_id_b_1610612754',
     'team_id_b_1610612755',
     'team_id_b_1610612756',
     'team_id_b_1610612757',
     'team_id_b_1610612758',
     'team_id_b_1610612759',
     'team_id_b_1610612760',
     'team_id_b_1610612761',
     'team_id_b_1610612762',
     'team_id_b_1610612763',
     'team_id_b_1610612764',
     'team_id_b_1610612765',
     'team_id_b_1610612766',
     'season_2010-11',
     'season_2011-12',
     'season_2012-13',
     'season_2013-14',
     'season_2014-15',
     'season_2015-16',
     'season_2016-17',
     'season_2017-18',
     'game_time_zone_a_America/Chicago',
     'game_time_zone_a_America/Denver',
     'game_time_zone_a_America/Detroit',
     'game_time_zone_a_America/Indiana/Indianapolis',
     'game_time_zone_a_America/Los_Angeles',
     'game_time_zone_a_America/New_York',
     'game_time_zone_a_America/Phoenix',
     'game_time_zone_a_America/Toronto',
     'time_zone_a_America/Chicago',
     'time_zone_a_America/Denver',
     'time_zone_a_America/Detroit',
     'time_zone_a_America/Indiana/Indianapolis',
     'time_zone_a_America/Los_Angeles',
     'time_zone_a_America/New_York',
     'time_zone_a_America/Phoenix',
     'time_zone_a_America/Toronto',
     'city_proper_metro_Area_a_Atlanta',
     'city_proper_metro_Area_a_Boston',
     'city_proper_metro_Area_a_Charlotte',
     'city_proper_metro_Area_a_Chicago',
     'city_proper_metro_Area_a_Cleveland',
     'city_proper_metro_Area_a_Dallas–Fort Worth',
     'city_proper_metro_Area_a_Denver',
     'city_proper_metro_Area_a_Detroit',
     'city_proper_metro_Area_a_Houston',
     'city_proper_metro_Area_a_Indianapolis',
     'city_proper_metro_Area_a_Los Angeles',
     'city_proper_metro_Area_a_Memphis',
     'city_proper_metro_Area_a_Miami',
     'city_proper_metro_Area_a_Milwaukee',
     'city_proper_metro_Area_a_Minneapolis/St. Paul',
     'city_proper_metro_Area_a_New Orleans',
     'city_proper_metro_Area_a_New York',
     'city_proper_metro_Area_a_Oklahoma City',
     'city_proper_metro_Area_a_Orlando',
     'city_proper_metro_Area_a_Philadelphia',
     'city_proper_metro_Area_a_Phoenix',
     'city_proper_metro_Area_a_Portland',
     'city_proper_metro_Area_a_Sacramento',
     'city_proper_metro_Area_a_Salt Lake City',
     'city_proper_metro_Area_a_San Antonio',
     'city_proper_metro_Area_a_San Francisco',
     'city_proper_metro_Area_a_Toronto',
     'Country/Region_a_Canada',
     'Country/Region_a_United States',
     'game_time_zone_b_America/Chicago',
     'game_time_zone_b_America/Denver',
     'game_time_zone_b_America/Detroit',
     'game_time_zone_b_America/Indiana/Indianapolis',
     'game_time_zone_b_America/Los_Angeles',
     'game_time_zone_b_America/New_York',
     'game_time_zone_b_America/Phoenix',
     'game_time_zone_b_America/Toronto',
     'time_zone_b_America/Chicago',
     'time_zone_b_America/Denver',
     'time_zone_b_America/Detroit',
     'time_zone_b_America/Indiana/Indianapolis',
     'time_zone_b_America/Los_Angeles',
     'time_zone_b_America/New_York',
     'time_zone_b_America/Phoenix',
     'time_zone_b_America/Toronto',
     'city_proper_metro_Area_b_Boston',
     'city_proper_metro_Area_b_Charlotte',
     'city_proper_metro_Area_b_Chicago',
     'city_proper_metro_Area_b_Cleveland',
     'city_proper_metro_Area_b_Dallas–Fort Worth',
     'city_proper_metro_Area_b_Denver',
     'city_proper_metro_Area_b_Detroit',
     'city_proper_metro_Area_b_Houston',
     'city_proper_metro_Area_b_Indianapolis',
     'city_proper_metro_Area_b_Los Angeles',
     'city_proper_metro_Area_b_Memphis',
     'city_proper_metro_Area_b_Miami',
     'city_proper_metro_Area_b_Milwaukee',
     'city_proper_metro_Area_b_Minneapolis/St. Paul',
     'city_proper_metro_Area_b_New Orleans',
     'city_proper_metro_Area_b_New York',
     'city_proper_metro_Area_b_Oklahoma City',
     'city_proper_metro_Area_b_Orlando',
     'city_proper_metro_Area_b_Philadelphia',
     'city_proper_metro_Area_b_Phoenix',
     'city_proper_metro_Area_b_Portland',
     'city_proper_metro_Area_b_Sacramento',
     'city_proper_metro_Area_b_Salt Lake City',
     'city_proper_metro_Area_b_San Antonio',
     'city_proper_metro_Area_b_San Francisco',
     'city_proper_metro_Area_b_Toronto',
     'city_proper_metro_Area_b_Washington, D.C.',
     'Country/Region_b_Canada',
     'Country/Region_b_United States']
    
    return column_name_list_172_trial_4_ordered
















def get_top_50_feature_importances_data_frame_train_test_collection_of_model_172_trial_4(series_random_forest_feature_importances,
                                                                                         df_172_trial_4_indicator_variables_collection_train_test):

    column_name_list_172_trial_4_top_50 = series_random_forest_feature_importances[0:50].index.to_list()


    df_172_trial_4_indicator_variables_top_50_collection_train_test = {}

    #get train top 50 features of model 172 trial 4 (with indicator variables)
    df_172_trial_4_indicator_variables_top_50_collection_train_test['train'] = \
    df_172_trial_4_indicator_variables_collection_train_test['train'].loc[:, column_name_list_172_trial_4_top_50]

    #get test top 50 features of model 172 trial 4 (with indicator variables)
    df_172_trial_4_indicator_variables_top_50_collection_train_test['test'] = \
    df_172_trial_4_indicator_variables_collection_train_test['test'].loc[:, column_name_list_172_trial_4_top_50]


    df_172_trial_4_indicator_variables_top_50_collection_train_test['train_test'] = pd.concat([df_172_trial_4_indicator_variables_top_50_collection_train_test['train'],
                                                                                         df_172_trial_4_indicator_variables_top_50_collection_train_test['test']])
    
    return df_172_trial_4_indicator_variables_top_50_collection_train_test




def get_top_50_features_data_frame_train_test_collection_of_model_161_trial_1(df_time_series_cross_validation_index_8_train,
                                                                              df_time_series_cross_validation_index_8_test,
                                                                              column_name_list_161_trial_1_top_50_features):
    
    df_161_top_50_collection_train_test = {}

    df_161_top_50_collection_train_test['train'] = df_time_series_cross_validation_index_8_train.loc[:, column_name_list_161_trial_1_top_50_features]

    df_161_top_50_collection_train_test['test'] = df_time_series_cross_validation_index_8_test.loc[:, column_name_list_161_trial_1_top_50_features]

    df_161_top_50_collection_train_test['train_test'] = pd.concat([df_161_top_50_collection_train_test['train'], df_161_top_50_collection_train_test['test']])

    return df_161_top_50_collection_train_test



def random_forest_regressor_model_180_time_series_cross_validation_column_name_list_ordered():
    column_name_list_model_180_ordered = ['latitude_diff',
                                          'longitude_diff',
                                          'games_back_to_back_percentage_game_window999_diff',
                                          'city_proper_metro_Area_b_San Antonio',
                                          'future_Away_game_percent_day_window999_diff',
                                          'future_home_game_percent_day_window999_diff',
                                          'away_game_percent_day_window999_diff',
                                          'games_back_to_back_count_game_window999_diff',
                                          'home_game_percent_day_window999_diff',
                                          'future_Away_game_percent_day_window12_diff',
                                          'game_count_day_window999_diff',
                                          'future_home_game_percent_day_window12_diff',
                                          'games_back_to_back_percentage_game_window12_diff',
                                          'city_proper_metro_Area_b_Oklahoma City',
                                          'city_proper_metro_Area_b_Sacramento',
                                          'season_2011-12',
                                          'home_game_percent_day_window12_diff',
                                          'game_date_diff_diff',
                                          'future_home_game_count_day_window12_diff',
                                          'away_game_percent_day_window12_diff',
                                          'team_id_a_1610612751',
                                          'future_game_count_day_window999_diff',
                                          'home_game_count_day_window999_diff',
                                          'future_Away_game_percent_day_window8_diff',
                                          'away_game_percent_day_window8_diff',
                                          'future_home_game_percent_day_window8_diff',
                                          'home_game_percent_day_window8_diff',
                                          'away_game_count_day_window999_diff',
                                          'game_count_day_window12_diff',
                                          'future_game_count_day_window12_diff',
                                          'home_game_count_day_window8_diff',
                                          'home_game_count_day_window12_diff',
                                          'games_back_to_back_percentage_game_window5_diff',
                                          'away_game_count_day_window12_diff',
                                          'city_proper_metro_Area_a_San Francisco',
                                          'team_id_a_1610612747',
                                          'future_game_count_day_window8_diff',
                                          'future_Away_game_count_day_window8_diff',
                                          'games_back_to_back_count_game_window12_diff',
                                          'game_count_day_window8_diff',
                                          'game_count_day_window5_diff',
                                          'future_game_count_day_window2_diff',
                                          'away_game_count_day_window2_diff',
                                          'city_proper_metro_Area_b_Philadelphia',
                                          'NET_RATING_cma999_diff',
                                          'away_home_a',
                                          'PIE_cma999_diff',
                                          'strength_of_schedule_diff',
                                          'opp_losses_diff',
                                          'PIE_cma12_diff',
                                          'E_NET_RATING_cma12_diff',
                                          'away_home__cma999_diff',
                                          'pf_cma999_diff',
                                          'fg3a_cma12_diff',
                                          'E_NET_RATING_cmax5_diff',
                                          'lc_max_MIN_cma5_diff',
                                          'lc_max_POSS_cma5_diff',
                                          'spread_cma12_diff',
                                          'lc_max_MIN_cma12_diff',
                                          'opp_opps_win_pct_diff',
                                          'w_pct_cma12_diff',
                                          'EFG_PCT_cma999_diff',
                                          'TS_PCT_cma999_diff',
                                          'lc_sum_POSS_cmax5_diff',
                                          'E_OFF_RATING_cma999_diff',
                                          'lc_sum_TS_PCT_cma999_diff',
                                          'PIE_cma5_diff',
                                          'E_DEF_RATING_cma12_diff',
                                          'w_pct_cma5_diff',
                                          'lc_mean_TS_PCT_cma5_diff',
                                          'AST_RATIO_cmax5_diff',
                                          'lc_mean_EFG_PCT_cmax999_diff',
                                          'w_pct_cmax5_diff',
                                          'wins_diff',
                                          'lc_max_AST_TOV_cmax999_diff',
                                          'lc_sum_AST_RATIO_cma5_diff',
                                          'OFF_RATING_cma999_diff',
                                          'lc_sum_TS_PCT_cma5_diff',
                                          'NET_RATING_cma12_diff',
                                          'pf_cmax999_diff',
                                          'lc_sum_NET_RATING_cmax999_diff',
                                          'lc_max_AST_TOV_cmax12_diff',
                                          'lc_sum_AST_PCT_cmax5_diff',
                                          'fg3_pct_cma999_diff',
                                          'ft_pct_cmax12_diff',
                                          'lc_max_MIN_cmax5_diff',
                                          'E_NET_RATING_cma5_diff',
                                          'E_NET_RATING_cmax12_diff']
    return column_name_list_model_180_ordered





def combine_drop_and_order_model_172_trial_4_model_161_trial_1_top_50_features(series_random_forest_feature_importances_172_trial_4,
                                                                               df_172_trial_4_indicator_variables_collection_train_test,
                                                                               df_time_series_cross_validation_index_8_train,
                                                                               df_time_series_cross_validation_index_8_test,
                                                                               column_name_list_161_trial_1_top_50_features):
    
    #get top50 feature importances data frame train test collection of model 172 trial 4
    df_172_trial_4_indicator_variables_top_50_collection_train_test = \
    get_top_50_feature_importances_data_frame_train_test_collection_of_model_172_trial_4(series_random_forest_feature_importances_172_trial_4,
                                                                                             df_172_trial_4_indicator_variables_collection_train_test)


    #get top 50 features train test collection of model 161 trial 1 
    df_161_top_50_collection_train_test = get_top_50_features_data_frame_train_test_collection_of_model_161_trial_1(df_time_series_cross_validation_index_8_train, df_time_series_cross_validation_index_8_test, column_name_list_161_trial_1_top_50_features)

    #combine top 50 features from model and 161 and 172 
    df_random_forest_time_series_cross_validation_9_splits = pd.concat([df_161_top_50_collection_train_test['train_test'],
                                                                        df_172_trial_4_indicator_variables_top_50_collection_train_test['train_test']], axis=1)

    #drop duplicate like features
    column_name_list_drop_duplicate_like = ['spread_cma999_diff',
                                            'E_NET_RATING_cma999_diff',
                                            'opp_win_pct_diff',
                                            'win_pct_diff',
                                            'travel_distance_diff',
                                            'team_id_b_1610612759',
                                            'team_id_b_1610612760',
                                            'team_id_a_1610612744',
                                            'team_id_b_1610612758',
                                            'team_id_b_1610612755',
                                            'losses_diff',
                                            'opp_wins_diff']
    
    column_name_list_model_180_ordered = random_forest_regressor_model_180_time_series_cross_validation_column_name_list_ordered()

    df_random_forest_time_series_cross_validation_9_splits = df_random_forest_time_series_cross_validation_9_splits.drop(columns=column_name_list_drop_duplicate_like).loc[:, column_name_list_model_180_ordered]
    
    return df_random_forest_time_series_cross_validation_9_splits






















#################################################################################################################################

def get_trained_random_forst_regressor_180_trial_6_time_series_cross_validation_9_splits_collection(df_random_forest_time_series_cross_validation_9_splits, df_score_difference_a_2010_2018):

    #build
    from sklearn.model_selection import cross_validate
    from sklearn.ensemble import RandomForestRegressor


    random_forest_regressor = RandomForestRegressor(n_estimators=200,
                                                    min_samples_split=100,
                                                    max_depth=10,
                                                    random_state=47,
                                                    criterion = 'mae')
    
    
    from sklearn.model_selection import TimeSeriesSplit
    time_series_cross_validation_9_splits = TimeSeriesSplit(max_train_size=None, n_splits=9)

    model_180_random_forest_tscv9_trial_6 = cross_validate(random_forest_regressor, df_random_forest_time_series_cross_validation_9_splits, df_score_difference_a_2010_2018, cv=time_series_cross_validation_9_splits.split(df_random_forest_time_series_cross_validation_9_splits), 
                                                           return_estimator=True)
    return model_180_random_forest_tscv9_trial_6



#################################################################################################################################


#################################################################################################################################

def get_score_difference_a_predicted(df_random_forest_time_series_cross_validation_9_splits, model_180_random_forest_tscv9_trial_6):
    
    #get data frame test collection
    from sklearn.model_selection import TimeSeriesSplit
    time_series_cross_validation_9_splits = TimeSeriesSplit(max_train_size=None, n_splits=9)
    df_test_collection = {}

    for i, (train_indices, test_indices) in enumerate(time_series_cross_validation_9_splits.split(df_random_forest_time_series_cross_validation_9_splits)):
        df_test_collection[i] = df_random_forest_time_series_cross_validation_9_splits.iloc[test_indices, :]
        #print('Train: %s | test: %s' % (train_indices, test_indices))
    
    
    
    #get prediction from random forest regressor collection
    train_time_series_cross_validation_9_splits_test_predictions_collection = {}

    for i in range(len(model_180_random_forest_tscv9_trial_6.get('estimator'))):
        train_time_series_cross_validation_9_splits_test_predictions_collection[i] = model_180_random_forest_tscv9_trial_6.get('estimator')[i].predict(df_test_collection[i])
        
    #convert prediction collection to concatenated list
    train_time_series_cross_validation_9_splits_test_predictions_list = list([])
    for i in range(9):
        train_time_series_cross_validation_9_splits_test_predictions_list += list(train_time_series_cross_validation_9_splits_test_predictions_collection[i])


    #convert prediction list to data frame
    df_time_series_cross_validation_9_splits_score_difference_a_predicted = pd.DataFrame({'spread_a_predicted':train_time_series_cross_validation_9_splits_test_predictions_list})
    
    return df_time_series_cross_validation_9_splits_score_difference_a_predicted


#################################################################################################################################









################################################################################################################################

def get_gradient_boosting_regressor_grid_search_cross_validation(X_train,
                                                                 y_train,
                                                                 X_test,
                                                                 y_test):
    

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, learning_curve
    from sklearn.pipeline import make_pipeline
    from sklearn import __version__ as sklearn_version
    import datetime
    import pickle

    gradient_boosting_regressor=GradientBoostingRegressor()

    parameter_grid_dict={'n_estimators':[20, 50, 100], 'learning_rate':[.001,0.01,.1], 'max_depth':[1,2,4], 'subsample':[.5,.75,1], 'random_state':[1]}

    
    gradient_boosting_regressor_grid_search_cross_validation=GridSearchCV(estimator=gradient_boosting_regressor, param_grid=parameter_grid_dict, scoring='neg_mean_absolute_error', n_jobs=-1, cv=5)

    gradient_boosting_regressor_grid_search_cross_validation.fit(X_train, y_train)
    
    return gradient_boosting_regressor_grid_search_cross_validation

#################################################################################################################################











#################################################################################################################################
def get_random_forest_regressor_time_series_cross_validation_9_splits_decision_tree_prediction_bootstrap_means(df_random_forest_time_series_cross_validation_9_splits, model_180_random_forest_tscv9_trial_6):
    
    #get decision tree prediction for each random forest regresssor of 200 estimators
    from sklearn.model_selection import TimeSeriesSplit
    time_series_cross_validation_9_splits = TimeSeriesSplit(max_train_size=None, n_splits=9)

    Y_train_tscv9_test_decision_tree_predictions_collection = {} 

    Y_train_tscv9_test_decision_tree_predictions = pd.DataFrame({})
    for i, (train_indices, test_indices) in enumerate(time_series_cross_validation_9_splits.split(df_random_forest_time_series_cross_validation_9_splits)):
        for pred in model_180_random_forest_tscv9_trial_6.get('estimator')[i].estimators_:
            temp = pd.Series(pred.predict(df_random_forest_time_series_cross_validation_9_splits.iloc[test_indices,:]))
            Y_train_tscv9_test_decision_tree_predictions = pd.concat([Y_train_tscv9_test_decision_tree_predictions, temp], axis=1)
        Y_train_tscv9_test_decision_tree_predictions_collection[i] = Y_train_tscv9_test_decision_tree_predictions
        Y_train_tscv9_test_decision_tree_predictions = pd.DataFrame({})



    #convert decision tree prediction collection to list
    df_list = [Y_train_tscv9_test_decision_tree_predictions_collection[i] for i in range(len(Y_train_tscv9_test_decision_tree_predictions_collection))]



    #convert decision tree prediction list to data frame 
    Y_train_tscv9_test_decision_tree_predictions = pd.concat(df_list)



    #get bootstrap means of decision tree prediction per game
    np.random.seed(965)
    n_replicas = 10000

    Y_train_tscv9_test_bootstrap_means = \
    [list(Y_train_tscv9_test_decision_tree_predictions.sample(n=8, replace=True, axis=1).mean(axis=1).values) for i in range(n_replicas)]


    #convert bootstrap means list to data frame
    df_Y_train_tscv9_test_bootstrap_means = pd.DataFrame(Y_train_tscv9_test_bootstrap_means).T




    #rename bootstrap mean columns
    Y_train_tscv9_test_bootstrap_means_column_names_list = list(df_Y_train_tscv9_test_bootstrap_means.columns)


    Y_train_tscv9_test_bootstrap_means_column_names_value_list = \
    ['bootstrap_mean' + str(k) for k in Y_train_tscv9_test_bootstrap_means_column_names_list]

    Y_train_tscv9_test_bootstrap_means_column_names_dict = \
    dict(zip(Y_train_tscv9_test_bootstrap_means_column_names_list, Y_train_tscv9_test_bootstrap_means_column_names_value_list))


    df_Y_train_tscv9_test_bootstrap_means.rename(columns=Y_train_tscv9_test_bootstrap_means_column_names_dict, inplace=True)



    #round bootstrap means to 2 decimal places
    df_Y_train_tscv9_test_bootstrap_means = df_Y_train_tscv9_test_bootstrap_means.round(2)
    
    return df_Y_train_tscv9_test_bootstrap_means

#################################################################################################################################







#################################################################################################################################

def add_score_difference_a_score_difference_a_predicted_game_id_game_date_team_id_a_team_id_b_away_home_a_to_bootstrap_means(df_time_series_cross_validation_9_splits_test_game_id_game_date_team_id_a_team_id_b_away_home_a_score_difference_a,
                                                                                                                             df_time_series_cross_validation_9_splits_score_difference_a_predicted,
                                                                                                                             df_time_series_cross_validation_9_splits_test_random_forest_regressor_decision_tree_prediction_bootstrap_means):

    #add score difference a, score differnce a predicted, game_id, game_date, team_id_a, team_id_b, and away_home_a to bootstrap means


    #add test score difference a predictions column
    df_time_series_cross_validation_9_splits_test_score_difference_a_score_difference_a_predicted = \
    pd.concat([df_time_series_cross_validation_9_splits_test_game_id_game_date_team_id_a_team_id_b_away_home_a_score_difference_a, 
               df_time_series_cross_validation_9_splits_score_difference_a_predicted], 
              axis=1)


    #add bootstrap mean columns
    df_time_series_cross_validation_9_splits_test_random_forest_decision_tree_score_difference_a_prediction_bootstrap_means = \
    pd.concat([df_time_series_cross_validation_9_splits_test_score_difference_a_score_difference_a_predicted, 
              df_time_series_cross_validation_9_splits_test_random_forest_regressor_decision_tree_prediction_bootstrap_means], 
              axis=1)

    return df_time_series_cross_validation_9_splits_test_random_forest_decision_tree_score_difference_a_prediction_bootstrap_means




#################################################################################################################################






#################################################################################################################################

def combine_score_difference_a_score_difference_a_predicted_bootstrap_means_with_sportsbook_spread_price(df_time_series_cross_validation_9_splits_test_sportsbook_spread_first, df_time_series_cross_validation_9_splits_test_random_forest_decision_tree_score_difference_a_prediction_bootstrap_means, df_time_series_cross_validation_9_splits_test_random_forest_regressor_decision_tree_prediction_bootstrap_means):

    #get data frame from merge of (away) team_id and team_id_a
    df_team_id_a_sportsbook_bootstrap_means = pd.merge(df_time_series_cross_validation_9_splits_test_sportsbook_spread_first.rename(columns={'team_id':'team_id_a'}), 
                                                       df_time_series_cross_validation_9_splits_test_random_forest_decision_tree_score_difference_a_prediction_bootstrap_means, 
                                                       on=['game_id', 'team_id_a'])

    df_team_id_a_sportsbook_bootstrap_means = \
    df_team_id_a_sportsbook_bootstrap_means.rename(columns={'score_difference_a':'score_difference', 
                                                            'spread_a_predicted':'score_difference_predicted'})

    df_team_id_a_sportsbook_bootstrap_means = df_team_id_a_sportsbook_bootstrap_means.drop(columns=['team_id_b'])


    #get data frame from merge of (away) team_id and team_id_b  and flip sign of score_difference, score_difference_predicted, and bootstrap means
    df_team_id_b_sportsbook_bootstrap_means = pd.merge(df_time_series_cross_validation_9_splits_test_sportsbook_spread_first.rename(columns={'team_id':'team_id_b'}), 
                                                       df_time_series_cross_validation_9_splits_test_random_forest_decision_tree_score_difference_a_prediction_bootstrap_means, 
                                                       on=['game_id', 'team_id_b'])

    df_team_id_b_sportsbook_bootstrap_means.loc[:, 'score_difference_a'] = df_team_id_b_sportsbook_bootstrap_means.score_difference_a*-1
    df_team_id_b_sportsbook_bootstrap_means.loc[:, 'spread_a_predicted'] = df_team_id_b_sportsbook_bootstrap_means.spread_a_predicted*-1

    column_name_list_bootstrap_means = df_time_series_cross_validation_9_splits_test_random_forest_regressor_decision_tree_prediction_bootstrap_means.columns.to_list()
    df_team_id_b_sportsbook_bootstrap_means.loc[:, column_name_list_bootstrap_means] = df_team_id_b_sportsbook_bootstrap_means.loc[:, column_name_list_bootstrap_means]*-1


    df_team_id_b_sportsbook_bootstrap_means = \
    df_team_id_b_sportsbook_bootstrap_means.rename(columns={'score_difference_a':'score_difference', 
                                                            'spread_a_predicted':'score_difference_predicted'})

    df_team_id_b_sportsbook_bootstrap_means = df_team_id_b_sportsbook_bootstrap_means.drop(columns=['team_id_a'])



    #combine data frames df_team_id_a and df_team_id_b that used (away) team_id to merge and fix score_difference, score_difference_predicted, bootstrap_mean sign 
    df_spread1_score_difference_score_difference_predicted_bootstrap_means = \
    pd.concat([df_team_id_a_sportsbook_bootstrap_means.rename(columns={'team_id_a':'team_id'}), 
               df_team_id_b_sportsbook_bootstrap_means.rename(columns={'team_id_b':'team_id'})])

    df_spread1_score_difference_score_difference_predicted_bootstrap_means = \
    df_spread1_score_difference_score_difference_predicted_bootstrap_means.sort_values(['game_date', 'game_id'])

    df_spread1_score_difference_score_difference_predicted_bootstrap_means = \
    df_spread1_score_difference_score_difference_predicted_bootstrap_means.reset_index(drop=True)

    return df_spread1_score_difference_score_difference_predicted_bootstrap_means



#################################################################################################################################




#################################################################################################################################

def extract_bet_on_spread1_spread2_column_based_on_score_difference_predicted(df):
    df.loc[df.loc[:, 'score_difference_predicted'] < df.loc[:, 'spread1'], 'bet_spread1_spread2'] = 'spread1'

    df.loc[df.loc[:, 'score_difference_predicted'] > df.loc[:, 'spread1'], 'bet_spread1_spread2'] = 'spread2'
    
    df.loc[df.loc[:, 'score_difference_predicted'] == df.loc[:, 'spread1'], 'bet_spread1_spread2'] = 'neither'
    
    return df

#################################################################################################################################




#################################################################################################################################

def extract_winning_bet_column_based_on_spread1_spread2_and_score_difference(df):
    df.loc[(df.loc[:, 'bet_spread1_spread2'] == 'spread1') & (df.loc[:, 'spread1'] > df.loc[:, 'score_difference']), 'winning_bet'] = 'yes'

    df.loc[(df.loc[:, 'bet_spread1_spread2'] == 'spread1') & (df.loc[:, 'spread1'] < df.loc[:, 'score_difference']), 'winning_bet'] = 'no'
    
    df.loc[(df.loc[:, 'bet_spread1_spread2'] == 'spread2') & (df.loc[:, 'spread1'] < df.loc[:, 'score_difference']), 'winning_bet'] = 'yes'
    
    df.loc[(df.loc[:, 'bet_spread1_spread2'] == 'spread2') & (df.loc[:, 'spread1'] > df.loc[:, 'score_difference']), 'winning_bet'] = 'no'
    
    df.loc[df.loc[:, 'bet_spread1_spread2'] == 'neither', 'winning_bet'] = 'no bet placed'
    
    df.loc[(df.loc[:, 'spread1'] == df.loc[:, 'score_difference']), 'winning_bet'] = 'push'
    
    return df

#################################################################################################################################




#################################################################################################################################

def extract_price1_break_even_price2_break_even_price_break_even_from_price1_price2_bet_spread1_spread2(df):

    df.loc[df.loc[:, 'price1'] > 0, 'price1_break_even'] = 100 / (100 + abs(df.loc[df.loc[:, 'price1'] > 0, 'price1']))

    df.loc[df.loc[:, 'price1'] < 0, 'price1_break_even'] = abs(df.loc[df.loc[:, 'price1'] < 0, 'price1']) / (abs(df.loc[df.loc[:, 'price1'] < 0, 'price1']) + 100)


    df.loc[df.loc[:, 'price2'] > 0, 'price2_break_even'] = 100 / (100 + abs(df.loc[df.loc[:, 'price2'] > 0, 'price2']))

    df.loc[df.loc[:, 'price2'] < 0, 'price2_break_even'] = abs(df.loc[df.loc[:, 'price2'] < 0, 'price2']) / (abs(df.loc[df.loc[:, 'price2'] < 0, 'price2']) + 100)


    df.loc[df.loc[:, 'bet_spread1_spread2'] == 'spread1', 'price_break_even'] = df.loc[:, 'price1_break_even']

    df.loc[df.loc[:, 'bet_spread1_spread2'] == 'spread2', 'price_break_even'] = df.loc[:, 'price2_break_even']
    
    return df

#################################################################################################################################


#################################################################################################################################
def extract_price1_winning_bet_yes_roi_price2_winning_bet_yes_roi_price_winning_bet_yes_roi_from_price1_price2(df):

    df.loc[df.loc[:, 'price1'] > 0, 'price1_winning_bet_yes_roi'] = df.loc[df.loc[:, 'price1'] > 0, 'price1'] / 100

    df.loc[df.loc[:, 'price1'] < 0, 'price1_winning_bet_yes_roi'] = 100 / abs(df.loc[df.loc[:, 'price1'] < 0, 'price1'])

    df.loc[df.loc[:, 'price2'] > 0, 'price2_winning_bet_yes_roi'] = df.loc[df.loc[:, 'price2'] > 0, 'price2'] / 100

    df.loc[df.loc[:, 'price2'] < 0, 'price2_winning_bet_yes_roi'] = 100 / abs(df.loc[df.loc[:, 'price2'] < 0, 'price2'])
    
    df.loc[df.loc[:, 'bet_spread1_spread2'] == 'spread1', 'price_winning_bet_yes_roi'] = df.loc[:, 'price1_winning_bet_yes_roi']

    df.loc[df.loc[:, 'bet_spread1_spread2'] == 'spread2', 'price_winning_bet_yes_roi'] = df.loc[:, 'price2_winning_bet_yes_roi']
    
    
    return df
#################################################################################################################################






#################################################################################################################################

def extract_bet_roi(df):

    df.loc[(df.loc[:, 'winning_bet'] == 'push'), 'bet_roi'] = 0
    df.loc[(df.loc[:, 'winning_bet'] == 'no'), 'bet_roi'] = -1
    df.loc[(df.loc[:, 'winning_bet'] == 'yes'), 'bet_roi'] = df.loc[(df.loc[:, 'winning_bet'] == 'yes'), 'price_winning_bet_yes_roi']
    
    return df

#################################################################################################################################



#################################################################################################################################
def extract_score_difference_predicted_spread1_diff_score_difference_predicted_spread1_diff_abs(df):

    df.loc[:, 'score_difference_predicted_spread1_diff'] = df.loc[:, 'score_difference_predicted'] - df.loc[:, 'spread1']

    df.loc[:, 'score_difference_predicted_spread1_diff_abs'] = abs(df.loc[:, 'score_difference_predicted'] - df.loc[:, 'spread1'])
    
    return df
#################################################################################################################################






#################################################################################################################################
def extract_spread1_abs_score_difference_predicted_abs(df):
    df.loc[:, 'spread1_abs'] = abs(df.loc[:, 'spread1'])

    df.loc[:, 'score_difference_predicted_abs'] = abs(df.loc[:, 'score_difference_predicted'])
    
    return df

#################################################################################################################################






#################################################################################################################################

def drop_unneeded_intermediate_sportsbook_feature_engineering_columns(df):
    return df.drop(columns=['price1_break_even', 'price2_break_even',
                            'spread2',
                            'price1', 'price2',
                            'price1_winning_bet_yes_roi', 'price2_winning_bet_yes_roi', 'price_winning_bet_yes_roi'])

#################################################################################################################################


#################################################################################################################################
def reorder_column_names_of_sportbook_betting_featured(df):
    column_name_list = list(df.columns)

    #get boostrap mean columns
    column_name_list_bootstrap_mean = [k for k in column_name_list if 'bootstrap_mean' in k]

    #reorder columns name list
    column_name_list_sportsbook_betting_features_reordered = ['game_id',
                                                              'team_id',
                                                              'a_team_id',
                                                              'game_date',
                                                              'spread1',
                                                              'spread1_abs',
                                                              'score_difference_predicted',
                                                              'score_difference_predicted_abs',
                                                              'score_difference_predicted_spread1_diff',
                                                              'score_difference_predicted_spread1_diff_abs',
                                                              'score_difference',
                                                              'bet_spread1_spread2',
                                                              'price_break_even',
                                                              'winning_bet',
                                                              'bet_roi'] + column_name_list_bootstrap_mean

    #reorder data frame column names
    df = df.loc[:, column_name_list_sportsbook_betting_features_reordered]
    
    return df
#################################################################################################################################












#################################################################################################################################

def extract_features_spread1_abs_score_difference_predicted_abs_score_difference_predicted_spread1_diff_score_difference_predicted_spread1_diff_abs_price_break_even_winning_bet_bet_roi(df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet):
    
    #extract price1 break even, price2 break even, price break even
    df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even = \
    extract_price1_break_even_price2_break_even_price_break_even_from_price1_price2_bet_spread1_spread2(df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet)

    #extract price1_winning_bet_yes_roi, price2_winning_bet_yes_roi, price_winning_bet_yes_roi

    df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_price_winning_bet_yes_roi = \
    extract_price1_winning_bet_yes_roi_price2_winning_bet_yes_roi_price_winning_bet_yes_roi_from_price1_price2(df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even)


    #extract bet_roi
    df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_price_winning_bet_yes_roi_bet_roi = \
    extract_bet_roi(df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_price_winning_bet_yes_roi)

    #extract score_difference_predicted_spread1_diff, score_difference_predicted_spread1_diff_abs
    df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_price_winning_bet_yes_roi_bet_roi = \
    extract_score_difference_predicted_spread1_diff_score_difference_predicted_spread1_diff_abs(df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_price_winning_bet_yes_roi_bet_roi)

    #extract spread1_abs, score_difference_predicted_abs
    df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_price_winning_bet_yes_roi_bet_roi = \
    extract_spread1_abs_score_difference_predicted_abs(df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_price_winning_bet_yes_roi_bet_roi)


    #drop unneeded intermediate sportsbook columns
    df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi = \
    drop_unneeded_intermediate_sportsbook_feature_engineering_columns(df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_price_winning_bet_yes_roi_bet_roi)


    #reorder data frame columns of sportbook features
    df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi = \
    reorder_column_names_of_sportbook_betting_featured(df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi)


    #drop rows to games not bet on
    df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi = \
    df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi.loc[df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi.winning_bet != 'no bet placed', :]

    return df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi


#################################################################################################################################















#################################################################################################################################

def get_df_bootstrap_mean_number_from_df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi(df):
    column_name_list_bootstrap_mean_number = [k for k in list(df.columns) if 'bootstrap_mean' in k]
    df_bootstrap_mean_number = df.loc[:, column_name_list_bootstrap_mean_number]
    
    return df_bootstrap_mean_number, column_name_list_bootstrap_mean_number

#################################################################################################################################






#################################################################################################################################
def replace_bootstrap_mean_number_with_percent_confidence_interval_place_bet_yes_no(df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi):

    #get df_bootstrap_mean_number and column_name_list_bootstrap_mean_number
    df_bootstrap_mean_number, column_name_list_bootstrap_mean_number = get_df_bootstrap_mean_number_from_df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi(df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi)



    #create number percent confidence interval lower limit and upper limit list

    #create list with numbers from 0 to 100
    zero_to_one_hundred_list = [k for k in range(101)]

    lower_limit_upper_limit_list = ['_lower_limit', '_upper_limit']
    percent_confidence_interval_lower_limit_upper_limit_list = [str(number) + '_percent_confidence_interval' + lower_upper for number in zero_to_one_hundred_list for lower_upper in lower_limit_upper_limit_list]


    #create percent confidence interval lower quantile and upper quantile list
    zero_one_hundred_list = [0, 100]
    plus_one_minus_one_list = [1, -1]
    confidence_interval_lower_quantile_upper_quantile_list = [(zero_one_hundred + plus_one_minus_one*((100 - percent_confidence_interval) / 2)) / 100 for percent_confidence_interval in zero_to_one_hundred_list for zero_one_hundred, plus_one_minus_one in zip(zero_one_hundred_list, plus_one_minus_one_list)]


    #calculate the boostrap mean number quantiles
    df_bootstrap_mean_number_quantity_quantile = pd.DataFrame({})
    for quantile in confidence_interval_lower_quantile_upper_quantile_list:
        df_bootstrap_mean_number_quantity_quantile = pd.concat([df_bootstrap_mean_number_quantity_quantile, df_bootstrap_mean_number.quantile(q=quantile, axis=1)], axis=1)


    #rename to percent confidence interval lower limit and upper limit from bootstrap mean number quantity quantile

    #build column name dictionary
    column_name_list_confidence_interval_lower_quantile_upper_quantile = df_bootstrap_mean_number_quantity_quantile.columns.to_list()
    lower_upper_confidence_interval_quantile_dict = dict(zip(column_name_list_confidence_interval_lower_quantile_upper_quantile, percent_confidence_interval_lower_limit_upper_limit_list))

    #rename columns and give new data frame name
    lower_upper_confidence_interval_quantile_dict[0.5] = '0_percent_confidence_interval_lower_limit_upper_limit'

    df_bootstrap_mean_confidence_interval_lower_limit_upper_limit = df_bootstrap_mean_number_quantity_quantile.rename(columns=lower_upper_confidence_interval_quantile_dict)
    del df_bootstrap_mean_number_quantity_quantile

    df_bootstrap_mean_confidence_interval_lower_limit_upper_limit.columns.values[0] = '0_percent_confidence_interval_lower_limit'
    df_bootstrap_mean_confidence_interval_lower_limit_upper_limit.columns.values[1] = '0_percent_confidence_interval_upper_limit'


    def replace_df_bootstrap_mean_number_with_df_prediction_percent_confidence_interval_upper_limit_lower_limit(df, column_name_list_bootstrap_mean_number, df_bootstrap_mean_confidence_interval_lower_limit_upper_limit):
        df = df.drop(columns=column_name_list_bootstrap_mean_number)
        return pd.concat([df, df_bootstrap_mean_confidence_interval_lower_limit_upper_limit], 
                         axis=1)


    df_spread1_score_difference_score_difference_predicted_bet_spread1_spread2_winning_bet_price_break_even_bet_roi_prediction_percent_confidence_interval_upper_limit_lower_limit = \
    replace_df_bootstrap_mean_number_with_df_prediction_percent_confidence_interval_upper_limit_lower_limit(df_spread1_score_difference_score_difference_predicted_bootstrap_means_bet_spread1_spread2_winning_bet_price_break_even_bet_roi,
                                                                                                            column_name_list_bootstrap_mean_number,
                                                                                                            df_bootstrap_mean_confidence_interval_lower_limit_upper_limit)


    def extract_feature_percent_confidence_interval_place_bet_yes_no(df,
                                                                     percent_confidence_interval_lower_limit_upper_limit_list):
        it = iter(percent_confidence_interval_lower_limit_upper_limit_list)
        for x in it:

            column_name_first = x
            column_name_second = next(it)

            #print(str(column_name_first) + ' and ' + str(column_name_second))

            column_name_new = str(x.split('_')[0]) + '_percent_confidence_interval_place_bet_yes_no'

            df.loc[:, column_name_new] = 'no'

            df.loc[(df.loc[:, column_name_first] > df.loc[:, 'spread1']) &
                   (df.loc[:, column_name_second] > df.loc[:, 'spread1']), column_name_new] = 'yes'
            df.loc[(df.loc[:, column_name_first] < df.loc[:, 'spread1']) &
                   (df.loc[:, column_name_second] < df.loc[:, 'spread1']), column_name_new] = 'yes'

        return df

    df_spread1_score_difference_score_difference_predicted_bet_spread1_spread2_winning_bet_price_break_even_bet_roi_prediction_percent_confidence_interval_upper_limit_lower_limit = \
    extract_feature_percent_confidence_interval_place_bet_yes_no(df_spread1_score_difference_score_difference_predicted_bet_spread1_spread2_winning_bet_price_break_even_bet_roi_prediction_percent_confidence_interval_upper_limit_lower_limit,
                                                                 percent_confidence_interval_lower_limit_upper_limit_list)


    #drop percent confidence interval upper lower columns
    def drop_percent_confidence_interval_lower_upper_columns(df):
        return df.drop(columns=percent_confidence_interval_lower_limit_upper_limit_list)

    df_spread1_score_difference_score_difference_predicted_bet_spread1_spread2_winning_bet_price_break_even_bet_roi_prediction_percent_confidence_interval_place_bet_yes_no = \
    drop_percent_confidence_interval_lower_upper_columns(df_spread1_score_difference_score_difference_predicted_bet_spread1_spread2_winning_bet_price_break_even_bet_roi_prediction_percent_confidence_interval_upper_limit_lower_limit)

    return df_spread1_score_difference_score_difference_predicted_bet_spread1_spread2_winning_bet_price_break_even_bet_roi_prediction_percent_confidence_interval_place_bet_yes_no

#################################################################################################################################















#################################################################################################################################

def get_df_time_series_cross_validation_9_splits_test_game_id_collection(df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b):
    df_game_id = df_team_advanced_box_scores_team_box_scores_collection_stacked_away_home_a_b['a_b'].loc[:, ['game_id']]

    time_series_cross_validation_9_splits = TimeSeriesSplit(max_train_size=None, n_splits=9)
    df_time_series_cross_validation_9_splits_test_game_id_collection = {}

    for i, (train_indices, test_indices) in enumerate(time_series_cross_validation_9_splits.split(df_game_id)):
        df_time_series_cross_validation_9_splits_test_game_id_collection[i] = df_game_id.iloc[test_indices, :]

    return df_time_series_cross_validation_9_splits_test_game_id_collection

#################################################################################################################################





#################################################################################################################################

def get_df_time_series_cross_validation_9_splits_folds_123456789_test_betting_policy_feature_collection(df,
                                                                                        df_time_series_cross_validation_9_splits_test_game_id_collection):
    df_time_series_cross_validation_9_splits_test_betting_policy_feature_collection = {}

    for i in range(len(df_time_series_cross_validation_9_splits_test_game_id_collection)):
        df_time_series_cross_validation_9_splits_test_betting_policy_feature_collection[i] = pd.merge(df, df_time_series_cross_validation_9_splits_test_game_id_collection[i], on='game_id')

    return df_time_series_cross_validation_9_splits_test_betting_policy_feature_collection

#################################################################################################################################


#################################################################################################################################
def get_concat_of_first_n_of_test_collection(df_collection, n=7):
    df_time_series_cross_validation_9_splits_test_1234567 = pd.DataFrame()
    for i in range(n):
        df_time_series_cross_validation_9_splits_test_1234567 = pd.concat([df_time_series_cross_validation_9_splits_test_1234567, df_collection[i]])

    df_time_series_cross_validation_9_splits_test_1234567 = df_time_series_cross_validation_9_splits_test_1234567.reset_index(drop=True)
    return df_time_series_cross_validation_9_splits_test_1234567
#################################################################################################################################








##############################################################################################################################

'''Betting Policy: Overview'''

##############################################################################################################################


def plot_time_series_cross_validation_9_splits_with_learn_heuristic_and_test_heuristic(df):
    def get_time_series_cross_validation_9_splits_split_size(df):

        df = df.reset_index(drop=True)

        tscv9 = TimeSeriesSplit(max_train_size=None, n_splits=9)
        df_indices_collection = {}

        df_plot = pd.DataFrame({})

        for i, (train_indices, test_indices) in enumerate(tscv9.split(df)):
            train_indices_len = len(train_indices)
            test_indices_len = len(test_indices)

            df_temp = pd.DataFrame({'train_size':[train_indices_len], 'test_size':[test_indices_len]})
            df_plot = pd.concat([df_plot, df_temp])

            df_indices_collection[i] = df.iloc[test_indices, :]

        df_plot = df_plot.reset_index(drop=True)
        df_plot = df_plot.reset_index().rename(columns={'index':'tscv9_index'})


        return df_plot

    df_plot = get_time_series_cross_validation_9_splits_split_size(df)

    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Patch


    df_plot.loc[:, 'total'] = df_plot.loc[:, 'train_size'].values + df_plot.loc[:, 'test_size']


    column_names_list = list(df_plot.columns)

    fold_name_list = ['fold' + str(i) for i in reversed(range(10))]


    fold_index_right_list = df_plot.loc[:, 'train_size'].to_list() + [10266]
    fold_index_right_list.reverse()

    zeros_array = np.zeros([10, 10])

    for i in range(10):
            zeros_array[i][0] = fold_index_right_list[i]
    fold_array = zeros_array

    fold_dict = dict(zip(fold_name_list, fold_array))

    df_folds = pd.DataFrame(fold_dict)

    df_plot_one_row = df_plot.iloc[[1],:]
    df_plot_one_row.values[:] = 0


    df_plot = pd.concat([df_plot_one_row, df_plot]).reset_index(drop=True)


    df_plot.loc[0, 'tscv9_index'] = 'folds'

    df_train_index0 = pd.DataFrame({'train_index0':[0]+[1032 for i in range(9)]})

    df_plot = pd.concat([df_plot , df_folds, df_train_index0], axis=1)


    df_plot_089 = df_plot.loc[[0,8,9],:]
    df_plot_089.loc[8, 'tscv9_index'] = 'learn heuristic'
    df_plot_089.loc[9, 'tscv9_index'] = 'test heuristic'
    df_plot_089.loc[8, 'train_size']

    df_plot_089 = pd.concat([df_plot_089, pd.DataFrame({'total_validation' : [0, 9240, 0]}, index=[0,8,9] )], axis=1)
    df_plot_089 = pd.concat([df_plot_089 , pd.DataFrame({'total_test' : [0, 0, 10266]}, index=[0,8,9])], axis=1)


    def before_bars():
        sns.set_theme(style="whitegrid")

        f, ax = plt.subplots(figsize=(15, 2))
        
        sns.set_color_codes("pastel")

        return f, ax

    def not_bars(number=None):

        handles, labels = ax.get_legend_handles_labels()

        handles = [handles[2], handles[0],  handles[1],]
        labels = [labels[2], labels[0], labels[1], ]


        ax.legend(ncol=1, 
                  loc="center right", 
                  frameon=True,
                  labels=labels,
                  handles=handles)


        ax.set(xlim=(0, 11000), 
               ylabel="time series cross validation index",
               xlabel="indices")

        sns.despine(left=True, 
                    bottom=True)

        p.set_title('Learn and Test Heuristic', fontsize=15)
        
        #save plot
        plt.savefig('figures/figure_time_series_cross_validation_9_splits_train_validation_test_folds_learn_test_heuristic' + str(number) +'.png', bbox_inches='tight', dpi=600)


    #######

    f, ax = before_bars()

    t = sns.barplot(x="total", 
                    y="tscv9_index", 
                    data=df_plot, 
                    color="b",
                    orient='h',
                    alpha=0)


    s = sns.barplot(x="total_validation", 
                    y="tscv9_index", 
                    data=df_plot_089,
                    label="validation", 
                    color="b",
                    orient='h',
                    alpha=1)

    s = sns.barplot(x="total_test", 
                    y="tscv9_index", 
                    data=df_plot_089,
                    label="test", 
                    color="paleturquoise",
                    orient='h',
                    alpha=1)


    #create physical graphing of totals with alpha 0, for next section reference.

    #next section used physical graph

    for i in reversed(range(10)):
        sns.set_color_codes("dark")
        ss = sns.barplot(x="fold" + str(i), 
                         y="tscv9_index", 
                         data=df_plot,
                         color="b",
                         orient='h')

    for i, p in enumerate(t.patches):
        if i < 10:
            if(i == 0):
                s.annotate(format(i, '.0f')+"", 
                           (p.get_x() + p.get_width() + 500, p.get_height() - 1.1), 
                           ha = 'center', va = 'center', 
                           size=15,
                           xytext = (0, -12), 
                           textcoords = 'offset points',
                           color='white')
            else:
                s.annotate(format(i, '.0f')+"", 
                           (p.get_x() + p.get_width() - 500, p.get_height() - 1.1), 
                           ha = 'center', va = 'center', 
                           size=15,
                           xytext = (0, -12), 
                           textcoords = 'offset points',
                           color='white')


    sns.set_color_codes("muted")

    sns.barplot(x="train_size", 
                y="tscv9_index", 
                data=df_plot_089,
                label="train", 
                color="b",
                orient='h')



    p = sns.barplot(x="train_index0", 
                    y="tscv9_index", 
                    data=df_plot_089,
                    label="None", 
                    color="lightgray",
                    orient='h',
                    alpha=.9)


    not_bars(number=3)


    
#################################################################################################################################




def plot_spread_bootstrap_mean_distribution_by_game_id(df, game_id, confidence_interval=95, plot_name='single_game_score_difference_bookstrap_mean_distribution', legend_position_upper_left_upper_right='upper right'):
       
    bootstrap_mean_column_names_list = [k for k in df.columns if 'bootstrap_mean' in k]
    
    df_game_id_T = df.loc[(df.loc[:, 'game_id'] == game_id), bootstrap_mean_column_names_list].T
    
    median = df_game_id_T.median().values[0]

    (100 - confidence_interval) / 2

    
    lower = np.percentile(df_game_id_T, (100 - confidence_interval) / 2)
    upper = np.percentile(df_game_id_T, confidence_interval + (100 - confidence_interval) / 2)

    
    sns.set_theme(style="whitegrid")    

    
    p = sns.distplot(df_game_id_T)
    
    max_height = p.get_lines()[0].get_data()[1].max()
    
    
    p.set_xlabel("Home - Away Score (PTS)", fontsize = 15)
    p.set_ylabel("Density", fontsize = 15)
    p.set_title('Single Game Score Difference Bootstrap Means', fontsize=15)


    score_difference = df.loc[(df.loc[:, 'game_id'] == game_id), 'score_difference'].values[0]
    spread1_book = df.loc[(df.loc[:, 'game_id'] == game_id), 'spread1'].values[0]

    score_difference_line = plt.axvline(score_difference, color='blue', linestyle='--')
    spread1_book_line = plt.axvline(spread1_book, color='gold', linestyle='--')
    
    plt.text(score_difference + .1, p.get_ylim()[1] - .12,'home - away score', rotation=90)
    plt.text(spread1_book + .1, max_height - .05,'spread1 book', rotation=90)
    
    color = None
   

    p.grid(False)

    
    if spread1_book >= lower and spread1_book <= upper:  
        Rectangle = plt.gca().add_patch(plt.Rectangle((lower, 0),upper - lower, max_height + 1, fill=True, linewidth=1, fc='gray', alpha=.3))
        color = 'gray'

        handles = [score_difference_line, spread1_book_line, Rectangle]

        
    elif (median > spread1_book and spread1_book > score_difference) or (median < spread1_book and spread1_book < score_difference):  
        Rectangle = plt.gca().add_patch(plt.Rectangle((lower, 0),upper - lower, max_height + 1, fill=True, linewidth=1, fc='red', alpha=.3))
        color = 'red'

        handles = [score_difference_line, spread1_book_line, Rectangle]

        
        
    elif (median > spread1_book and score_difference > spread1_book) or (median < spread1_book and score_difference < spread1_book):  
        Rectangle = plt.gca().add_patch(plt.Rectangle((lower, 0),upper - lower, max_height + 1, fill=True, linewidth=1, fc='green', alpha=.3))
        color = 'green'
        
        handles = [score_difference_line, spread1_book_line, Rectangle]

        
    else:
        Rectangle = plt.gca().add_patch(plt.Rectangle((lower, 0), upper - lower, max_height + 1, fill=True, linewidth=1, fc='purple', alpha=.3))
        color = 'purple'
        handles = [score_difference_line, spread1_book_line, Rectangle]

        
    plt.legend(handles = handles,
           labels=['home - away score', 'spread1 book', str(confidence_interval) + '% CI'],
           loc=legend_position_upper_left_upper_right,
           fontsize=10)


    if plot_name != None and color == 'red':
        plt.savefig('figures/figure_single_game_score_difference_bookstrap_mean_distribution' + '_losing_' + str(confidence_interval) + '_CI_game_id_' + str(game_id) + '.png', bbox_inches='tight', dpi=1000)
    elif plot_name != None and color == 'green':
        plt.savefig('figures/figure_single_game_score_difference_bookstrap_mean_distribution' + '_winning_' + str(confidence_interval) + '_CI_game_id_' + str(game_id) + '.png', bbox_inches='tight', dpi=1000)
    elif plot_name != None and color == 'gray':
        plt.savefig('figures/figure_single_game_score_difference_bookstrap_mean_distribution' + '_psuedo_no_bet_' + str(confidence_interval) + '_CI_game_id_' + str(game_id) + '.png', bbox_inches='tight', dpi=1000)
    


################################################################################################################################



#################################################################################################################################
'''Betting Policy: Learn Heuristic and Test Heuristic'''
#################################################################################################################################

def fix_price_break_even_upper_limit(df):
    df.loc[:, 'price_break_even_upper_limit'] = df.loc[:, 'price_break_even_upper_limit'].replace({1:0.5939})
    return df


def filter_data_frame_by_betting_policy_result(df,
                                               games_bet_on_percentage_lower_limit=1.99,
                                               games_bet_on_winning_bet_percentage_lower_limit=49.9999):
    df = df.loc[df.loc[:, 'games_bet_on_percentage'] > games_bet_on_percentage_lower_limit, :]
    return df.loc[df.loc[:, 'games_bet_on_winning_bet_percentage'] > games_bet_on_winning_bet_percentage_lower_limit, :]


#################################################################################################################################

'''Betting Policy: Learn Heuristic'''

#################################################################################################################################

def roi_min_max_final_and_graph(df, 
                                percentage_of_account_balance_to_bet, 
                                principal=1000, 
                                return_true_false = False,
                                return_graph_true_false = False,
                                base_or_alternative = 'base'):
    
    if (return_graph_true_false == True) and (base_or_alternative == 'base'):

        df = df.reset_index(drop=True)
        account_balance = principal
        
        bet_amount_bet_outcome_account_balance_roi = [[0, 0, account_balance, 0]]
        
        for i in range(df.shape[0]):
            
            if account_balance == None:
                print('account_balance is None')
            if percentage_of_account_balance_to_bet == None:
                print('percentage_of_account_balance_to_bet is None')
            
            bet_amount = account_balance * percentage_of_account_balance_to_bet
            
            bet_outcome = bet_amount * df.loc[i, 'bet_roi']

            account_balance  = account_balance + bet_outcome

            roi  = (account_balance - principal) / principal
            
            bet_amount_bet_outcome_account_balance_roi = bet_amount_bet_outcome_account_balance_roi + [[bet_amount, bet_outcome, account_balance, roi]]
        
        df_temp = pd.DataFrame(bet_amount_bet_outcome_account_balance_roi, 
                               columns=['bet_amount', 'bet_outcome', 'account_balance', 'roi'])

        largest_bet_gain = df_temp.loc[:, 'bet_outcome'].max()
        largest_bet_loss = df_temp.loc[:, 'bet_outcome'].min()
        
        roi_min = df_temp.loc[:, 'roi'].min()
        roi_max = df_temp.loc[:, 'roi'].max()
        
        if df.shape[0] > 0:
            roi_final = df_temp.loc[:, 'roi'].tail(1).values[0]
        else:
            roi_final = 0        
        
        print('roi_min: ' + str(roi_min * 100) + '%')
        print('roi_max: ' + str(round(roi_max, 2) * 100) + '%')
        print(color.BOLD + 'roi_final: ' + str(round(roi_final, 2) * 100) + '%' + color.END)
        print()

        plt.plot(df_temp.loc[:, 'roi'].reset_index(drop=True) * 100)
        plt.xlabel('Game Bet On Count')
        plt.ylabel('ROI (%)')
        plt.title('ROI vs Game Bet On Count')
        plt.savefig('figures/figure_time_series_cross_validation_9_splits_fold_9_ROI_chart_percentage_betting_scheme' + \
                    str(percentage_of_account_balance_to_bet)[2:] + \
                    '.png', dpi=900, tight_layout=True)
        plt.show()
                
        return [round(largest_bet_gain, 2), round(largest_bet_loss, 2), round(roi_min, 2), round(roi_max, 2), round(roi_final, 2)]
    elif (return_graph_true_false == True) and (base_or_alternative == 'alternative'):

        df = df.reset_index(drop=True)
        account_balance = principal
        
        bet_amount_bet_outcome_account_balance_roi = [[0, 0, account_balance, 0]]
        
        #build list of bet_amount, bet_outcome, account_balance, and roi
        for i in range(df.shape[0]):
            
            bet_amount = account_balance * percentage_of_account_balance_to_bet / 100
            
            bet_outcome = bet_amount * df.loc[i, 'bet_roi']

            account_balance  = account_balance + bet_outcome

            roi  = (account_balance - principal)  / principal
            
            bet_amount_bet_outcome_account_balance_roi = bet_amount_bet_outcome_account_balance_roi + [[bet_amount, bet_outcome, account_balance, roi]]
        
        #convert list to data frame
        df_temp = pd.DataFrame(bet_amount_bet_outcome_account_balance_roi, 
                               columns=['bet_amount', 'bet_outcome', 'account_balance', 'roi'])

        #get betting policy stats
        largest_bet_gain = df_temp.loc[:, 'bet_outcome'].max()
        largest_bet_loss = df_temp.loc[:, 'bet_outcome'].min()
        
        roi_min = df_temp.loc[:, 'roi'].min()
        roi_max = df_temp.loc[:, 'roi'].max()
        
        if df.shape[0] > 0:
            roi_final = df_temp.loc[:, 'roi'].tail(1).values[0]
        else:
            roi_final = 0
        
        #print roi stats
        print('roi_min: ' + str(roi_min * 100) + '%')
        print('roi_max: ' + str(round(roi_max * 100, 2)) + '%')
        print(color.BOLD + 'roi_final: ' + str(round(roi_final * 100, 2)) + '%' + color.END)
        print()

        #plot running roi per game bet on
        plt.plot(df_temp.loc[:, 'roi'].reset_index(drop=True)*100)
        plt.xlabel('Game Bet On Count')
        plt.ylabel('ROI (%)')
        plt.title('ROI vs Game Bet On Count')
        plt.savefig('figures/figure_time_series_cross_validation_9_splits_fold_9_ROI_chart_percentage_betting_scheme' + \
                    str(percentage_of_account_balance_to_bet)[2:] + \
                    '.png', dpi=900, tight_layout=True)
        plt.show()
                
        return [round(largest_bet_gain, 2), round(largest_bet_loss, 2), round(roi_min, 2), round(roi_max, 2), round(roi_final, 2)]

        
#         account_balance = principal
#         bet_size = principal / initial_min_number_of_bets_available_

#         df.loc[:, 'win_minus_loss_count'] = df.loc[:,'bet_roi'].rolling(99999, 0).sum()

#         df.loc[:, 'roi'] = df.loc[:, 'win_minus_loss_count'] / initial_min_number_of_bets_available * 100


#         roi_min = df.loc[:, 'roi'].min()
#         roi_max = df.loc[:, 'roi'].max()
#         roi_final = df.loc[:, 'roi'].tail(1).values[0]

#         class color:
#            BOLD = '\033[1m'
#            END = '\033[0m'

#         print('roi_min: ' + str(roi_min) + '%')
#         print('roi_max: ' + str(round(roi_max, 2)) + '%')
#         print(color.BOLD + 'roi_final: ' + str(round(roi_final, 2)) + '%' + color.END)
#         print()

#         plt.plot(df.loc[:, 'roi'].reset_index(drop=True))
#         plt.xlabel('Consecutive Game Bet On')
#         plt.ylabel('Cumulative ROI (%)')
#         plt.title('Cumulative ROI')
#         plt.show()
        
    if (return_true_false == True) and base_or_alternative == 'base':

        df = df.reset_index(drop=True)
        account_balance = principal
        
        bet_amount_bet_outcome_account_balance_roi = [[0, 0, account_balance, 0]]
        
        for i in range(df.shape[0]):
            
            bet_amount = account_balance * percentage_of_account_balance_to_bet
            
            bet_outcome = bet_amount * df.loc[i, 'bet_roi']

            account_balance  = account_balance + bet_outcome

            roi  = (account_balance - principal)  / principal
            
            bet_amount_bet_outcome_account_balance_roi = bet_amount_bet_outcome_account_balance_roi + [[bet_amount, bet_outcome, account_balance, roi]]
        
        df_temp = pd.DataFrame(bet_amount_bet_outcome_account_balance_roi, 
                               columns=['bet_amount', 'bet_outcome', 'account_balance', 'roi'])

        largest_bet_gain = df_temp.loc[:, 'bet_outcome'].max()
        largest_bet_loss = df_temp.loc[:, 'bet_outcome'].min()
        
        roi_min = df_temp.loc[:, 'roi'].min()
        roi_max = df_temp.loc[:, 'roi'].max()
        
        if df.shape[0] > 0:
            roi_final = df_temp.loc[:, 'roi'].tail(1).values[0]
        else:
            roi_final = 0
                 
        return [round(largest_bet_gain, 2), round(largest_bet_loss, 2), round(roi_min, 2), round(roi_max, 2), round(roi_final, 2)]
    
    elif (return_true_false == True) and (base_or_alternative == 'alternative'):

        df = df.reset_index(drop=True)
        account_balance = principal
        
        bet_amount_bet_outcome_account_balance_roi = [[0, 0, account_balance, 0]]
        
        for i in range(df.shape[0]):
            
            bet_amount = account_balance * percentage_of_account_balance_to_bet / 100
            
            bet_outcome = bet_amount * df.loc[i, 'bet_roi']

            account_balance  = account_balance + bet_outcome

            roi  = (account_balance - principal)  / principal
            
            bet_amount_bet_outcome_account_balance_roi = bet_amount_bet_outcome_account_balance_roi + [[bet_amount, bet_outcome, account_balance, roi]]
        
        df_temp = pd.DataFrame(bet_amount_bet_outcome_account_balance_roi, 
                               columns=['bet_amount', 'bet_outcome', 'account_balance', 'roi'])

        largest_bet_gain = df_temp.loc[:, 'bet_outcome'].max()
        largest_bet_loss = df_temp.loc[:, 'bet_outcome'].min()
        
        roi_min = df_temp.loc[:, 'roi'].min() * 100
        roi_max = df_temp.loc[:, 'roi'].max() * 100
        
        if df.shape[0] > 0:
            roi_final = df_temp.loc[:, 'roi'].tail(1).values[0] * 100
        else:
            roi_final = 0
                 
        return [round(largest_bet_gain, 2), round(largest_bet_loss, 2), round(roi_min, 2), round(roi_max, 2), round(roi_final, 2)]

    
    
    
    
    
def winning_bet_no_yes_max_streak(df, 
                                  percentage_of_account_balance_to_bet, 
                                  return_true_false = False):
    
    if return_true_false == False:
        df_winning_bet = df.loc[df.winning_bet != 'push', ['winning_bet']]


        df_winning_bet.loc[:, 'start_of_streak'] = df_winning_bet.loc[:, 'winning_bet'].ne(df_winning_bet.loc[:, 'winning_bet'].shift())
        df_winning_bet.loc[:, 'streak_id'] = df_winning_bet.loc[:, 'start_of_streak'].cumsum()
        df_winning_bet.loc[:, 'streak_counter'] = df_winning_bet.groupby('streak_id').cumcount() + 1

        df_streak_counter = df_winning_bet.reset_index(drop=True)

        df_streak_per_streak_id = df_streak_counter.groupby('streak_id')['streak_counter'].max().reset_index()
        df_streak_per_streak_id = df_streak_per_streak_id.rename(columns={'streak_counter':'streak_per_streak_id'})

        df_streak_counter_streak_per_id = pd.merge(df_streak_counter, df_streak_per_streak_id, on='streak_id')

        df_winning_bet_maxes = df_streak_counter_streak_per_id.groupby('winning_bet')['streak_per_streak_id'].max().reset_index()

        winning_bet_no_streak = df_winning_bet_maxes.loc[df_winning_bet_maxes.loc[:, 'winning_bet'] == 'no', 'streak_per_streak_id'].values[0]
        print('winning_bet_no_streak: ' + str(winning_bet_no_streak))

        winning_bet_yes_streak = df_winning_bet_maxes.loc[df_winning_bet_maxes.loc[:, 'winning_bet'] == 'yes', 'streak_per_streak_id'].values[0]

        print('winning_bet_yes_streak: ' + str(winning_bet_yes_streak))

    if return_true_false == True:
        return_list = []
        
        df_winning_bet = df.loc[df.loc[:, 'winning_bet'] != 'push', ['winning_bet']]


        df_winning_bet.loc[:, 'start_of_streak'] = df_winning_bet.loc[:, 'winning_bet'].ne(df_winning_bet.loc[:, 'winning_bet'].shift())
        df_winning_bet.loc[:, 'streak_id'] = df_winning_bet.loc[:, 'start_of_streak'].cumsum()
        df_winning_bet.loc[:, 'streak_counter'] = df_winning_bet.groupby('streak_id').cumcount() + 1

        df_streak_counter = df_winning_bet.reset_index(drop=True)

        df_streak_per_streak_id = df_streak_counter.groupby('streak_id')['streak_counter'].max().reset_index()
        df_streak_per_streak_id = df_streak_per_streak_id.rename(columns={'streak_counter':'streak_per_streak_id'})

        df_streak_counter_streak_per_id = pd.merge(df_streak_counter, df_streak_per_streak_id, on='streak_id')

        df_winning_bet_maxes = df_streak_counter_streak_per_id.groupby('winning_bet')['streak_per_streak_id'].max().reset_index()


        if df_winning_bet_maxes.loc[df_winning_bet_maxes.loc[:, 'winning_bet'] == 'no', 'streak_per_streak_id'].shape[0] > 0:
            winning_bet_no_streak = df_winning_bet_maxes.loc[df_winning_bet_maxes.loc[:, 'winning_bet'] == 'no', 'streak_per_streak_id'].values[0]
            return_list = return_list + [winning_bet_no_streak]
        else:
            winning_bet_no_streak = 0
            return_list = return_list + [winning_bet_no_streak]
            
        if df_winning_bet_maxes.loc[df_winning_bet_maxes.loc[:, 'winning_bet'] == 'yes', 'streak_per_streak_id'].shape[0] > 0:
            winning_bet_yes_streak = df_winning_bet_maxes.loc[df_winning_bet_maxes.loc[:, 'winning_bet'] == 'yes', 'streak_per_streak_id'].values[0]
            return_list = return_list + [winning_bet_yes_streak]
        else:
            winning_bet_yes_streak = 0
            return_list = return_list + [winning_bet_yes_streak]
        
        
        return return_list

    
    

def get_results_for_games_selected_by_policy(df, 
                                             percent_confidence_interval_place_bet_yes_no=None, 
                                             spread1_abs_lower_limit=None, 
                                             price_break_even_upper_limit=None,
                                             percentage_of_account_balance_to_bet = .05,
                                             principal = None,
                                             games_bet_on_lower_limit=None,
                                             games_bet_on_percentage_lower_limit=None,
                                             return_graph_true_false=False,
                                             roi_lower_limit=None):
    
    return_list = [percent_confidence_interval_place_bet_yes_no.split('_percent')[0], spread1_abs_lower_limit, price_break_even_upper_limit]
    
    
    if ((percent_confidence_interval_place_bet_yes_no != None) & (spread1_abs_lower_limit != None) & (price_break_even_upper_limit != None)):
        df_filtered = df.loc[(df.loc[:, percent_confidence_interval_place_bet_yes_no] == 'yes') &
                              (df.spread1_abs > spread1_abs_lower_limit) &
                              (df.price_break_even < price_break_even_upper_limit), :]
    
    elif ((percent_confidence_interval_place_bet_yes_no == None) & (spread1_abs_lower_limit != None) & (price_break_even_upper_limit != None)):
        df_filtered = df.loc[(df.spread1_abs > spread1_abs_lower_limit) &
                              (df.price_break_even < price_break_even_upper_limit), :]
        
    elif ((percent_confidence_interval_place_bet_yes_no != None) & (spread1_abs_lower_limit == None) & (price_break_even_upper_limit != None)):
        df_filtered = df.loc[(df.loc[:, percent_confidence_interval_place_bet_yes_no] == 'yes') &
                              (df.price_break_even < price_break_even_upper_limit), :]
        
    elif ((percent_confidence_interval_place_bet_yes_no != None) & (spread1_abs_lower_limit != None) & (price_break_even_upper_limit == None)):
        df_filtered = df.loc[(df.loc[:, percent_confidence_interval_place_bet_yes_no] == 'yes') &
                              (df.spread1_abs > spread1_abs_lower_limit), :]
        
    elif ((percent_confidence_interval_place_bet_yes_no == None) & (spread1_abs_lower_limit == None) & (price_break_even_upper_limit != None)):
        df_filtered = df.loc[(df.price_break_even < price_break_even_upper_limit), :]
        
    elif ((percent_confidence_interval_place_bet_yes_no != None) & (spread1_abs_lower_limit == None) & (price_break_even_upper_limit == None)):
        df_filtered = df.loc[(df.loc[:, percent_confidence_interval_place_bet_yes_no] == 'yes'), :]
        
    elif ((percent_confidence_interval_place_bet_yes_no == None) & (spread1_abs_lower_limit != None) & (price_break_even_upper_limit == None)):
        df_filtered = df.loc[(df.spread1_abs > spread1_abs_lower_limit), :]
    else:
        df_filtered = df
    
    row_count_df_filtered = df_filtered.shape[0]
    return_list = return_list + [row_count_df_filtered]
    
    row_count_df = df.shape[0]
    games_bet_on_percentage = row_count_df_filtered / row_count_df * 100
    return_list = return_list + [round(games_bet_on_percentage, 2)]
    
    
    if row_count_df_filtered > 0:
        games_bet_on_winning_bet_percentage = df_filtered.loc[df_filtered.winning_bet == 'yes', :].shape[0] / \
                                                                row_count_df_filtered * 100
        return_list = return_list + [round(games_bet_on_winning_bet_percentage, 2)]
    else:
        games_bet_on_winning_bet_percentage = 0
        return_list = return_list + [round(games_bet_on_winning_bet_percentage, 2)]

    
    if df_filtered.loc[df_filtered.winning_bet != 'push', :].shape[0] > 0:
        games_bet_on_winning_bet_percentage_push_adjusted = df_filtered.loc[df_filtered.winning_bet == 'yes', :].shape[0] / \
                                                                 df_filtered.loc[df_filtered.winning_bet != 'push', :].shape[0] * 100
        return_list = return_list + [round(games_bet_on_winning_bet_percentage_push_adjusted, 2)]
    else:
        games_bet_on_winning_bet_percentage_push_adjusted = 0
        return_list = return_list + [round(games_bet_on_winning_bet_percentage_push_adjusted, 2)]
        
    streaks_list = winning_bet_no_yes_max_streak(df_filtered, 
                                                  percentage_of_account_balance_to_bet, 
                                                  return_true_false=True)
    
    return_list = return_list + streaks_list
    
    largest_gain_largest_loss_roi_min_max_final_list = roi_min_max_final_and_graph(df_filtered, percentage_of_account_balance_to_bet = percentage_of_account_balance_to_bet, return_true_false=True, return_graph_true_false = False, base_or_alternative='base')

    return_list = return_list + largest_gain_largest_loss_roi_min_max_final_list
    
    df_returned = pd.DataFrame([return_list])
    
    return df_returned


#################################################################################################################################































#################################################################################################################################

'''Betting Policy: Test Heuristic'''

#################################################################################################################################


def get_return_list2(percentage_of_account_balance_to_bet,
                     percent_confidence_interval_place_bet_yes_no,
                     spread1_abs_lower_limit,
                     price_break_even_upper_limit,
                     row_count_df_filtered,
                     games_bet_on_percentage,
                     games_bet_on_winning_bet_percentage,
                     games_bet_on_winning_bet_percentage_push_adjusted,
                     streaks_list,
                     roi_min_max_final_list):
    return_list = []
    
    return_list += [percentage_of_account_balance_to_bet]
    
    if len(percent_confidence_interval_place_bet_yes_no) < 2:
        return_list += [percent_confidence_interval_place_bet_yes_no[0].split('_percent')[0]]
    else:
        return_list += [[k.split('_percent')[0] for k in percent_confidence_interval_place_bet_yes_no]]
    
    if len(spread1_abs_lower_limit) < 2:
        return_list += spread1_abs_lower_limit
    else:
        return_list += [spread1_abs_lower_limit]

    if len(price_break_even_upper_limit) < 2:
        return_list += price_break_even_upper_limit
    else:
        return_list += [price_break_even_upper_limit]
    
    return_list += [row_count_df_filtered, 
                    round(games_bet_on_percentage, 2), 
                    round(games_bet_on_winning_bet_percentage, 2), 
                    round(games_bet_on_winning_bet_percentage_push_adjusted, 2)]

    return_list += streaks_list + roi_min_max_final_list
    

    return return_list
    

    

    
def get_results_for_games_selected_by_policies2(df, 
                                                percent_confidence_interval_place_bet_yes_no = None, 
                                                spread1_abs_lower_limit = None, 
                                                price_break_even_upper_limit = None,
                                                percentage_of_account_balance_to_bet = 5,
                                                principal = None,
                                                games_bet_on_lower_limit = None,
                                                games_bet_on_percentage_lower_limit = None,
                                                return_graph_true_false=False,
                                                roi_lower_limit = None):

    policy_criteria_collection = get_policy_criteria_collection(percent_confidence_interval_place_bet_yes_no, 
                                                                spread1_abs_lower_limit, 
                                                                price_break_even_upper_limit)
    
    
    df_filtered = get_games_filtered_by_policies(df, 
                                                 policy_criteria_collection,
                                                 percent_confidence_interval_place_bet_yes_no,
                                                 spread1_abs_lower_limit,
                                                 price_break_even_upper_limit)

    
    row_count_df_filtered = df_filtered.shape[0]
    
    
    row_count_df = df.shape[0]
    games_bet_on_percentage = row_count_df_filtered / row_count_df * 100

    
    if row_count_df_filtered > 0:
        games_bet_on_winning_bet_percentage = df_filtered.loc[df_filtered.loc[:, 'winning_bet'] == 'yes', :].shape[0] / \
                                                                row_count_df_filtered * 100  
    else:
        games_bet_on_winning_bet_percentage = 0

    
    if df_filtered.loc[df_filtered.loc[:, 'winning_bet'] != 'push', :].shape[0] > 0:
        games_bet_on_winning_bet_percentage_push_adjusted = df_filtered.loc[df_filtered.winning_bet == 'yes', :].shape[0] / \
                                                                            df_filtered.loc[df_filtered.loc[:, 'winning_bet'] != 'push', :].shape[0] * 100
    else:
        games_bet_on_winning_bet_percentage_push_adjusted = 0

    
    streaks_list = winning_bet_no_yes_max_streak(df = df_filtered, 
                                                     percentage_of_account_balance_to_bet = percentage_of_account_balance_to_bet, 
                                                     return_true_false = True)
    
    
    roi_min_max_final_list = roi_min_max_final_and_graph(df = df_filtered, percentage_of_account_balance_to_bet = percentage_of_account_balance_to_bet, principal = principal, return_true_false = True, return_graph_true_false = return_graph_true_false, base_or_alternative='alternative')
    
    
    return pd.DataFrame(get_return_list2(percentage_of_account_balance_to_bet,
                                         percent_confidence_interval_place_bet_yes_no, 
                                         spread1_abs_lower_limit,
                                         price_break_even_upper_limit,
                                         row_count_df_filtered,
                                         games_bet_on_percentage,
                                         games_bet_on_winning_bet_percentage,
                                         games_bet_on_winning_bet_percentage_push_adjusted,
                                         streaks_list,
                                         roi_min_max_final_list)).T


#################################################################################################################################





#################################################################################################################################

def get_games_filtered_by_policy(df, 
                                 percent_confidence_interval_place_bet_yes_no, 
                                 spread1_abs_lower_limit, 
                                 price_break_even_upper_limit):
    
    if ((percent_confidence_interval_place_bet_yes_no != None) & (spread1_abs_lower_limit != None) & (price_break_even_upper_limit != None)):
        df_filtered = df.loc[(df.loc[:, percent_confidence_interval_place_bet_yes_no] == 'yes') &
                              (df.spread1_abs > spread1_abs_lower_limit) &
                              (df.price_break_even < price_break_even_upper_limit), :]

    elif ((percent_confidence_interval_place_bet_yes_no == None) & (spread1_abs_lower_limit != None) & (price_break_even_upper_limit != None)):
        df_filtered = df.loc[(df.spread1_abs > spread1_abs_lower_limit) &
                              (df.price_break_even < price_break_even_upper_limit), :]
        
    elif ((percent_confidence_interval_place_bet_yes_no != None) & (spread1_abs_lower_limit == None) & (price_break_even_upper_limit != None)):
        df_filtered = df.loc[(df.loc[:, percent_confidence_interval_place_bet_yes_no] == 'yes') &
                              (df.price_break_even < price_break_even_upper_limit), :]
        
    elif ((percent_confidence_interval_place_bet_yes_no != None) & (spread1_abs_lower_limit != None) & (price_break_even_upper_limit == None)):
        df_filtered = df.loc[(df.loc[:, percent_confidence_interval_place_bet_yes_no] == 'yes') &
                              (df.spread1_abs > spread1_abs_lower_limit), :]
        
    elif ((percent_confidence_interval_place_bet_yes_no == None) & (spread1_abs_lower_limit == None) & (price_break_even_upper_limit != None)):
        df_filtered = df.loc[(df.price_break_even < price_break_even_upper_limit), :]
        
    elif ((percent_confidence_interval_place_bet_yes_no != None) & (spread1_abs_lower_limit == None) & (price_break_even_upper_limit == None)):
        df_filtered = df.loc[(df.loc[:, percent_confidence_interval_place_bet_yes_no] == 'yes'), :]
        
    elif ((percent_confidence_interval_place_bet_yes_no == None) & (spread1_abs_lower_limit != None) & (price_break_even_upper_limit == None)):
        df_filtered = df.loc[(df.spread1_abs > spread1_abs_lower_limit), :]
    else:
        df_filtered = df
        
    return df_filtered
    


def get_policy_criteria_collection(percent_confidence_interval_place_bet_yes_no, 
                                   spread1_abs_lower_limit, 
                                   price_break_even_upper_limit):
    i = 0
    policy_criteria_collection = {}
    for a in percent_confidence_interval_place_bet_yes_no:
        for b in spread1_abs_lower_limit:
            for c in price_break_even_upper_limit:
                policy_criteria_collection[i] = [a, b, c]
                i = i + 1
    
    return policy_criteria_collection


def get_games_filtered_by_policies(df, 
                                   policy_criteria_collection,
                                   percent_confidence_interval_place_bet_yes_no,
                                   spread1_abs_lower_limit,
                                   price_break_even_upper_limit):

    n_policies = len(percent_confidence_interval_place_bet_yes_no) * len(spread1_abs_lower_limit) * len(price_break_even_upper_limit)

    df_filtered = pd.DataFrame()
    for i in range(n_policies):
        df_temp = get_games_filtered_by_policy(df, 
                                                percent_confidence_interval_place_bet_yes_no = policy_criteria_collection[i][0], 
                                                spread1_abs_lower_limit = policy_criteria_collection[i][1], 
                                                price_break_even_upper_limit = policy_criteria_collection[i][2])
        df_filtered = pd.concat([df_filtered, df_temp])
    
    df_filtered = df_filtered.sort_values(['game_date', 'game_id'])
    return df_filtered


def get_return_list(percent_confidence_interval_place_bet_yes_no, 
                    spread1_abs_lower_limit,
                    price_break_even_upper_limit,
                    row_count_df_filtered,
                    games_bet_on_percentage,
                    games_bet_on_winning_bet_percentage,
                    games_bet_on_winning_bet_percentage_push_adjusted,
                    streaks_list,
                    roi_min_max_final_list):
    return_list = []
    

    if len(percent_confidence_interval_place_bet_yes_no) < 2:
        return_list += [percent_confidence_interval_place_bet_yes_no[0].split('_percent')[0]]
    else:
        return_list += [[k.split('_percent')[0] for k in percent_confidence_interval_place_bet_yes_no]]
        

    if len(spread1_abs_lower_limit) < 2:
        return_list += spread1_abs_lower_limit
    else:
        return_list += [spread1_abs_lower_limit]

  
    if len(price_break_even_upper_limit) < 2:
        return_list += price_break_even_upper_limit
    else:
        return_list += [price_break_even_upper_limit]
      
    
    return_list += [row_count_df_filtered, 
                    round(games_bet_on_percentage, 2), 
                    round(games_bet_on_winning_bet_percentage, 2), 
                    round(games_bet_on_winning_bet_percentage_push_adjusted, 2)]
    

    return_list += streaks_list + roi_min_max_final_list

    
    return return_list
    


def get_results_for_games_selected_by_policies(df, 
                                               percent_confidence_interval_place_bet_yes_no = None, 
                                               spread1_abs_lower_limit = None, 
                                               price_break_even_upper_limit = None,
                                               percentage_of_account_balance_to_bet = .05,
                                               principal = None,
                                               games_bet_on_lower_limit = None,
                                               games_bet_on_percentage_lower_limit = None,
                                               return_graph_true_false = False,
                                               roi_lower_limit = None):

    policy_criteria_collection = get_policy_criteria_collection(percent_confidence_interval_place_bet_yes_no, 
                                                                spread1_abs_lower_limit, 
                                                                price_break_even_upper_limit)
    
    
    df_filtered = get_games_filtered_by_policies(df, 
                                                 policy_criteria_collection,
                                                 percent_confidence_interval_place_bet_yes_no,
                                                 spread1_abs_lower_limit,
                                                 price_break_even_upper_limit)

    
    row_count_df_filtered = df_filtered.shape[0]
    
    
    row_count_df = df.shape[0]
    games_bet_on_percentage = row_count_df_filtered / row_count_df * 100

    
    if row_count_df_filtered > 0:
        games_bet_on_winning_bet_percentage = df_filtered.loc[df_filtered.winning_bet == 'yes', :].shape[0] / \
                                                                row_count_df_filtered * 100  
    else:
        games_bet_on_winning_bet_percentage = 0

    
    if df_filtered.loc[df_filtered.winning_bet != 'push', :].shape[0] > 0:
        games_bet_on_winning_bet_percentage_push_adjusted = df_filtered.loc[df_filtered.loc[:, 'winning_bet'] == 'yes', :].shape[0] / \
                                                                 df_filtered.loc[df_filtered.loc[:, 'winning_bet'] != 'push', :].shape[0] * 100
    else:
        games_bet_on_winning_bet_percentage_push_adjusted = 0

    
    streaks_list = winning_bet_no_yes_max_streak(df = df_filtered, 
                                                     percentage_of_account_balance_to_bet = percentage_of_account_balance_to_bet, 
                                                     return_true_false = True)
    
    
    roi_min_max_final_list = roi_min_max_final_and_graph(df = df_filtered, percentage_of_account_balance_to_bet = percentage_of_account_balance_to_bet, principal = principal, return_true_false = True, return_graph_true_false=False, base_or_alternative='base')
    
    
    return pd.DataFrame(get_return_list(percent_confidence_interval_place_bet_yes_no, 
                                        spread1_abs_lower_limit,
                                        price_break_even_upper_limit,
                                        row_count_df_filtered,
                                        games_bet_on_percentage,
                                        games_bet_on_winning_bet_percentage,
                                        games_bet_on_winning_bet_percentage_push_adjusted,
                                        streaks_list,
                                        roi_min_max_final_list)).T





#################################################################################################################################
def index_policy_initial_conditions_and_results(df, column_name_list):

    initial_condition_result_list = ['initial conditions' for _ in range(4)] + ['results' for _ in range(11)]

    tuples = list(zip(*[initial_condition_result_list, column_name_list]))

    index = pd.MultiIndex.from_tuples(tuples)

    df = pd.DataFrame(df.values, index=index)

    df.columns = [' ']

    return df



#################################################################################################################################

'''Betting Policy Analysis'''

#################################################################################################################################

def get_non_negative_percentage(values_list):
    
    non_negative_count = 0
    for value in values_list:
        if value >= 0:
            non_negative_count += 1
    
    non_negative_percentage = non_negative_count / len(values_list) * 100
    
    return non_negative_percentage

def get_vertical_line_values(values_list,
                             alpha):
    
    values_list.sort()
    
    lower = np.percentile(values_list, (1-alpha) * 100 / 2)
    median = np.percentile(values_list, 50)
    upper = np.percentile(values_list, 100 - (1-alpha) * 100 / 2)
        
    return lower, median, upper

def get_values_list_from_collection(df_collection,
                                    column_name,
                                    number_range):

    values_list = []

    if 'roi' in column_name:
        for i in range(number_range):
            values_list += [df_collection[i].loc[:, column_name].values[0]]
        
    elif not 'roi' in column_name:
        for i in range(number_range):
            values_list += [df_collection[i].loc[:, column_name].values[0]]
            
    return values_list



def plot_histogram_of_bootstrap_dataframe(df_collection, 
                                          column_name='roi_final', 
                                          confidence_interval=None,
                                          title_name=None,
                                          xlabel=None,
                                          print_percentage_non_negative=False,
                                          vertical_line_at_percentile=None):

    values_list = get_values_list_from_collection(df_collection = df_collection,
                                                  column_name = column_name)
    
    
        
    if print_percentage_non_negative == True:
        non_negative_percentage = get_non_negative_percentage(values_list)
    
    
    if confidence_interval != None:
        
        lower, median, upper = get_vertical_line_values(values_list = values_list,
                                                        alpha = confidence_interval)
        
        empirical_confidence_interval = (round(lower, 2), round(upper, 2))
    
    elif vertical_line_at_percentile != None:
        vertical_line_roi = np.percentile(values_list, vertical_line_at_percentile)
    
        print('vertical_line_roi: ' + str(vertical_line_roi))

    sns.histplot(values_list, binwidth=1)
    #sns.histplot(values_list_, binwidth=1, kde=True)
    
    if title_name == None:
        plt.title(str(column_name) + ' Distribution')
    elif title_name != None:
        plt.title(title_name)
    
    if xlabel == None:
        plt.xlabel(str(column_name))
    elif xlabel != None:
        plt.xlabel(str(xlabel))

    if vertical_line_at_percentile == None:
        _ = plt.axvline(median, color='r')
        _ = plt.axvline(lower, color='r', linestyle='--')
        _ = plt.axvline(upper, color='r', linestyle='--')
    
    elif vertical_line_at_percentile != None:
        _ = plt.axvline(vertical_line_roi, color='r', linestyle='--')
    
    
    if vertical_line_at_percentile == None:
        print(str(confidence_interval * 100) + '%' +' confidence interval' + ': ' + str(empirical_confidence_interval))
    
    if print_percentage_non_negative == True:
        print('non-negative \'' + str(column_name) + '\' percentage' + ': ' + str(round(non_negative_percentage, 2)) + '%')
    
    
    if confidence_interval != None:
        plt.savefig(str(column_name) + 'distribution_' + str(int(confidence_interval_ * 100))  + \
                    '_percent_empirical_confidence_interval_percentage_betting_scheme_01.png', dpi=900, tight_layout=True)
    elif confidence_interval == None:
        plt.savefig(str(column_name) + 'distribution_vertical_line_at_percentile_' + str(int(vertical_line_at_percentile)) + '_ROI_'\
                    + str(int(vertical_line_roi))  + 'percent_percentage_betting_scheme_01.png', dpi=900, tight_layout=True)
#################################################################################################################################






def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape




