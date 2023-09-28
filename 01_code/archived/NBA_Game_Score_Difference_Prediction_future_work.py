

import pandas as pd
import os
import sqlite3
import numpy as np



def initialize_custom_notebook_settings():
    '''
    initialize the jupyter notebook display width'''
        
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))


    
'''
Convenience functions: read, sort, print, and save data frame.
'''
def p(df):
    '''
    Return the first 5 and last 5 rows of this DataFrame.'''
    if df.shape[0] > 6:
        print(df.shape)
        return pd.concat([df.head(), df.tail()])
    else:
        return df

def rcp(filename, parse_dates=None):
    '''
    Read a file from the processed_data folder.'''
    if parse_dates == None:
        return pd.read_csv(os.path.join('..', 'processed_data', filename))
    else:
        return pd.read_csv(os.path.join('..', 'processed_data', filename), parse_dates=parse_dates)
    
def rcr(filename, parse_dates=None):
    '''
    Read a file from the raw_data folder.'''
    if parse_dates == None:
        return pd.read_csv(os.path.join('..', 'raw_data', filename))
    else:
        return pd.read_csv(os.path.join('..', 'raw_data', filename), parse_dates=parse_dates)
    


def sr(df, column_list):
    '''
    Sort DataFrame by column(s) and reset its index.'''
    df = df.sort_values(column_list)
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



def save_and_return_data_frame(df,
                               filename,
                               index=False,
                               parse_dates=None):
    '''
    Save data frame function.'''
    
    datapath = os.path.join('..', 'processed_data')

    if not os.path.exists(datapath):
        os.mkdir(datapath)

    datapath_df = os.path.join(datapath, filename)
    if not os.path.exists(datapath_df):
        df.to_csv(datapath_df, index=index)
    elif os.path.exists(datapath_df):
        print('file already exists')
        
    return rcp(filename, parse_dates)


def show_data_frames_in_memory(dir_):
    alldfs = [var for var in dir_ if isinstance(eval(var), pd.core.frame.DataFrame)]

    print(alldfs)

    
    
    
    
    
    
    
    
    

'''
Data Wrangling functions
'''
def get_column_name_description():
    
    column_name_list = \
    ['game_id',
     'game_date',
     'TEAM_NAME_home',
     'TEAM_ABBREVIATION_home',
     'TEAM_CITY_home',
     'MIN_home',
     'E_OFF_RATING_home',
     'OFF_RATING_home',
     'E_DEF_RATING_home',
     'DEF_RATING_home',
     'E_NET_RATING_home',
     'NET_RATING_home',
     'AST_PCT_home',
     'AST_TOV_home',
     'AST_RATIO_home',
     'OREB_PCT_home',
     'DREB_PCT_home',
     'REB_PCT_home',
     'E_TM_TOV_PCT_home',
     'TM_TOV_PCT_home',
     'EFG_PCT_home',
     'TS_PCT_home',
     'USG_PCT_home',
     'E_USG_PCT_home',
     'E_PACE_home',
     'PACE_home',
     'PACE_PER40_home',
     'POSS_home',
     'PIE_home',
     'duration_home',
     'duration_minutes_home',
     'TEAM_NAME_away',
     'TEAM_ABBREVIATION_away',
     'TEAM_CITY_away',
     'MIN_away',
     'E_OFF_RATING_away',
     'OFF_RATING_away',
     'E_DEF_RATING_away',
     'DEF_RATING_away',
     'E_NET_RATING_away',
     'NET_RATING_away',
     'AST_PCT_away',
     'AST_TOV_away',
     'AST_RATIO_away',
     'OREB_PCT_away',
     'DREB_PCT_away',
     'REB_PCT_away',
     'E_TM_TOV_PCT_away',
     'TM_TOV_PCT_away',
     'EFG_PCT_away',
     'TS_PCT_away',
     'USG_PCT_away',
     'E_USG_PCT_away',
     'E_PACE_away',
     'PACE_away',
     'PACE_PER40_away',
     'POSS_away',
     'PIE_away',
     'duration_away',
     'duration_minutes_away',
     'matchup',
     'season_year',
     'season_type',
     'season',
     'team_id_away',
     'w_pct_away',
     'min_away',
     'fgm_away',
     'fga_away',
     'fg_pct_away',
     'fg3m_away',
     'fg3a_away',
     'fg3_pct_away',
     'ftm_away',
     'fta_away',
     'ft_pct_away',
     'oreb_away',
     'dreb_away',
     'reb_away',
     'ast_away',
     'stl_away',
     'blk_away',
     'tov_away',
     'pf_away',
     'pts_away',
     'team_id_home',
     'w_pct_home',
     'min_home',
     'fgm_home',
     'fga_home',
     'fg_pct_home',
     'fg3m_home',
     'fg3a_home',
     'fg3_pct_home',
     'ftm_home',
     'fta_home',
     'ft_pct_home',
     'oreb_home',
     'dreb_home',
     'reb_home',
     'ast_home',
     'stl_home',
     'blk_home',
     'tov_home',
     'pf_home',
     'pts_home',
     'wl_away',
     'wl_home',
     'score_difference_away',
     'score_difference_home']
    
    df_team_advanced_box_scores_away_home_team_box_scores_away_home_column_name = \
    pd.DataFrame({'column name':column_name_list})


    column_name_description_list = \
    ['unique indentifier for each nba game for the 2010-11 thru 2017-18 Regular Season and Playoffs',
     'year, month, and day of the nba game',
     'name of the nba away team',
     'three letter abbreviation of the nba away team',
     'city / region fo the nba away team',
     'minutes played cumulative ly bt the nba away team players',
     '(See MIN_away)',
     'name of the nba home team',
     'three letter abbreviation of the nba home team',
     'city / region fo the nba home team',
     'minutes played cumulatively by the nba home team players',
     '(See MIN_home)',
     '?estimated? offensive rating of the nba away team',
     'offensive rating of the nba away team',
     'points scored per 100 possessions of the nba away team',
     'points allowed per 100 posessions',
     '?estimated? net rating',
     'net rating = offensive rating - defensive rating; faster the pace, the more likely a higher net rating team is to beat a lower net rating team',
     'away team assits per field goal made',
     'assists to turnovers ratio of away team',
     'assist ratio; assists divided by uses; AST*100 / (FGA + [.44 * FTA] + AST + TOV)',
     'OREB / (OREB + OppDREB); percentage of team offensive rebounds grabbed by the away team',
     'DRB% = DRB  / (DRB + Opp. ORB); percentage of available defensive rebounds the away team obtains while on the floor',
     'percentage of available rebounds the away team grabbed while on the floor',
     'estimated turnover percentage',
     'away team turn over percentage; TeamTOV / (TeamFGA + (0.44 * TeamFTA) + TeamTOV',
     'effective field goal percentage; team evaluation metric that measures the effectiveness of 2-point shots and 3-point shots; (FGM + 0.5 * 3PM) / FGA; (2pt FGM + 1.5 * 3pt FGM) / FGA',
     'PTS / (2 * TSA) where True Shooting Attempts is TSA = FGA + 0.44 * FTA',
     'usage percentage; (FGA + Possession Ending FTA + TO) / POSS',
     '?estimated? usage percenteage of away team',
     '?estimated? pace of away team',
     'number of possessions per 48 minutes of the away team',
     'number of possessions per 40 minutes of the away team',
     'possessions; number of possessions played by a team. Please note: an Offensive Rebound does not create another possession, it simply makes the existing possession longer',
     'player impact estimate; (PTS + FGM + FTM - FGA - FTA + DREB + (.5 * OREB) + AST + STL + (.5 * BLK) - PF - TO) / (GmPTS + GmFGM + GmFTM - GmFGA - GmFTA + GmDREB + (.5 * GmOREB) + GmAST + GmSTL + (.5 * GmBLK) - GmPF - GmTO)',
     'number of minutes played by the away team',
     '?estimated? offensive rating of the nba home team',
     'offensive rating of the nba home team',
     'points scored per 100 possessions of the nba home team',
     'points allowed per 100 posessions',
     '?estimated? net rating',
     'net rating = offensive rating - defensive rating; faster the pace, the more likely a higher net rating team is to beat a lower net rating team',
     'home team assits per field goal made',
     'assists to turnovers ratio of home team',
     'assist ratio; assists divided by uses; AST*100 / (FGA + [.44 * FTA] + AST + TOV)',
     'OREB / (OREB + OppDREB); percentage of team offensive rebounds grabbed by the home team',
     'DRB% = DRB  / (DRB + Opp. ORB); percentage of available defensive rebounds the home team obtains while on the floor',
     'percentage of available rebounds the away team grabbed while on the floor',
     'estimated turnover percentage',
     'home team turn over percentage; TeamTOV / (TeamFGA + (0.44 * TeamFTA) + TeamTOV',
     'effective field goal percentage; team evaluation metric that measures the effectiveness of 2-point shots and 3-point shots; (FGM + 0.5 * 3PM) / FGA; (2pt FGM + 1.5 * 3pt FGM) / FGA',
     'PTS / (2 * TSA) where True Shooting Attempts is TSA = FGA + 0.44 * FTA',
     'usage percentage; (FGA + Possession Ending FTA + TO) / POSS',
     '?estimated? usage percenteage of home team',
     '?estimated? pace of home team',
     'number of possessions per 48 minutes of the home team',
     'number of possessions per 40 minutes of the home team',
     'possessions; number of possessions played by a team. Please note: an Offensive Rebound does not create another possession, it simply makes the existing possession longer',
     'player impact estimate; (PTS + FGM + FTM - FGA - FTA + DREB + (.5 * OREB) + AST + STL + (.5 * BLK) - PF - TO) / (GmPTS + GmFGM + GmFTM - GmFGA - GmFTA + GmDREB + (.5 * GmOREB) + GmAST + GmSTL + (.5 * GmBLK) - GmPF - GmTO)',
     'number of minutes played by the home team',
     'nba team matchup',
     'nba away team identity',
     'nba game won W or lost L by the away team',
     'win percentage of the away team',
     'nba game cumulative minutes played by nba away team players',
     'field goals made;number of field goals that a team has made. This includes both 2 pointers and 3 pointers',
     'field goals attempted; number of field goals that a team has attempted. This includes both 2 pointers and 3 pointers',
     'field goal percentage; percentage of field goal attempts that a team makes;  (FGM)/(FGA)',
     '3 point field goals made; number of 3 point field goals that a team has made',
     '3 point field goals attempted; number of 3 point field goals that a team has attempted',
     '3 point field goal percentage; percentage of 3 point field goal attempts that a team makes; (3PM)/(3PA)',
     'free throws made; number of free throws that a team has made',
     'free throws attempted; number of free throws that a team has attempted',
     'free throw percentage; percentage of free throw attempts that a team has made; (FTM)/(FTA)',
     'offensive rebounds; number of rebounds a team has collected while they were on offense',
     'defensive rebounds; number of rebounds a team has collected while they were on defense',
     'rebounds; a rebound occurs when a player recovers the ball after a missed shot. This statistic is the number of total rebounds a team has collected on either offense or defense',
     'assists; number of assists -- passes that lead directly to a made basket -- by a player (for a team)',
     'steals; number of times a defensive player or team takes the ball from a player on offense, causing a turnover (for a team)',
     'blocks; number of times a offensive player attempts a shot, and the defense player tips the ball, blocking their chance to score (for a team)',
     'turnovers; a turnover occurs when the player or team on offense loses the ball to the defense (for a team)',
     'personal fouls; number of personal fouls a team committed',
     'points; number of points scored (for a team)',

     'nba game won W or lost L by the home team',
     'nba home team identity',
     'win percentage of the home team',
     'nba game cumulative minutes played by nba home team players',
     'field goals made;number of field goals that a team has made. This includes both 2 pointers and 3 pointers',
     'field goals attempted; number of field goals that a team has attempted. This includes both 2 pointers and 3 pointers',
     'field goal percentage; percentage of field goal attempts that a team makes;  (FGM)/(FGA)',
     '3 point field goals made; number of 3 point field goals that a team has made',
     '3 point field goals attempted; number of 3 point field goals that a team has attempted',
     '3 point field goal percentage; percentage of 3 point field goal attempts that a team makes; (3PM)/(3PA)',
     'free throws made; number of free throws that a team has made',
     'free throws attempted; number of free throws that a team has attempted',
     'free throw percentage; percentage of free throw attempts that a team has made; (FTM)/(FTA)',
     'offensive rebounds; number of rebounds a team has collected while they were on offense',
     'defensive rebounds; number of rebounds a team has collected while they were on defense',
     'rebounds; a rebound occurs when a player recovers the ball after a missed shot. This statistic is the number of total rebounds a team has collected on either offense or defense',
     'assists; number of assists -- passes that lead directly to a made basket -- by a player (for a team)',
     'steals; number of times a defensive player or team takes the ball from a player on offense, causing a turnover (for a team)',
     'blocks; number of times a offensive player attempts a shot, and the defense player tips the ball, blocking their chance to score (for a team)',
     'turnovers; a turnover occurs when the player or team on offense loses the ball to the defense (for a team)',
     'personal fouls; number of personal fouls a team committed',
     'points; number of points scored (for a team)',           
     'Playoff or Regular Season game',
     'season years, e.g. 2010-11 season',
     'season start year']

    column_name_description_source_list = \
    ['NaN',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'https://www.basketball-reference.com/about/glossary.html',
     'https://www.basketball-reference.com/about/glossary.html',
     'NaN',
     'https://www.pivotanalysis.com/post/net-rating; https://paceandspacehoops.com/statistical-test-how-well-does-net-rating-correlate-with-wins/',
     'NaN',
     'https://www.teamrankings.com/nba/stat/assists-per-fgm',
     'NaN',
     'https://hoopshabit.com/2013/08/18/stat-central-understanding-strengths-shortcomings-of-assist-rate-metrics/',
     'https://www.nba.com/stats/help/faq/',
     'https://www.nba.com/resources/static/team/v2/thunder/statlab-activity-y3a5-1920-DRB.pdf; https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.pivotanalysis.com/post/what-is-turnover-percentage',
     'https://www.nbastuffer.com/analytics101/effective-field-goal-percentage-efg/; https://www.nba.com/bucks/features/boeder-120917; https://www.breakthroughbasketball.com/stats/effective-field-goal-percentage.html',
     'https://www.basketball-reference.com/about/glossary.html',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'NaN',
     'NaN',
     'https://www.nba.com/stats/help/glossary/',
     'https://justplaysolutions.com/analytics-academy/basketball-stats-glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'NaN',
     'https://www.basketball-reference.com/about/glossary.html',
     'https://www.basketball-reference.com/about/glossary.html',
     'NaN',
     'https://www.pivotanalysis.com/post/net-rating; https://paceandspacehoops.com/statistical-test-how-well-does-net-rating-correlate-with-wins/',
     'NaN',
     'https://www.teamrankings.com/nba/stat/assists-per-fgm',
     'NaN',
     'https://hoopshabit.com/2013/08/18/stat-central-understanding-strengths-shortcomings-of-assist-rate-metrics/',
     'https://www.nba.com/stats/help/faq/',
     'https://www.nba.com/resources/static/team/v2/thunder/statlab-activity-y3a5-1920-DRB.pdf; https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.pivotanalysis.com/post/what-is-turnover-percentage',
     'https://www.nbastuffer.com/analytics101/effective-field-goal-percentage-efg/; https://www.nba.com/bucks/features/boeder-120917; https://www.breakthroughbasketball.com/stats/effective-field-goal-percentage.html',
     'https://www.basketball-reference.com/about/glossary.html',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'NaN',
     'NaN',
     'https://www.nba.com/stats/help/glossary/',
     'https://justplaysolutions.com/analytics-academy/basketball-stats-glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'NaN',
     'NaN',
     'NaN',
     'NaN',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'https://www.nba.com/stats/help/glossary/',
     'NaN',
     'NaN',
     'NaN']  

    df_column_name_description = pd.DataFrame({'column_name_description':column_name_description_list, 
                                               'column_name_description_source':column_name_description_source_list})

    
    df_team_advanced_box_scores_away_home_team_box_scores_away_home_column_name_description = \
    pd.concat([df_team_advanced_box_scores_away_home_team_box_scores_away_home_column_name,
               df_column_name_description],
              axis=1)

    return df_team_advanced_box_scores_away_home_team_box_scores_away_home_column_name_description.reset_index(drop=True)





###########################################################################################################################

def get_team_advanced_box_score_stat_and_team_box_score_stat_12_game_moving_average_diff(df_team_advanced_box_scores_away_home_team_box_scores_away_home):
    


    def get_column_name_list_and_list_collection_from_team_advanced_box_scores_away_home_team_box_scores_away_home(
        df_team_advanced_box_scores_away_home_team_box_scores_away_home):
        '''
        get column name list collection from team advanced box scores and team box scores: column_name_list_collection_away_home_empty
        get column name list with suffix _away or _home: column_name_list_away_home'''
        
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



    def extract_and_add_columns_team_abbreviation_a_and_team_abbreviation_b_as_a_and_b(
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty):
        '''
        extract and add team abbreviation a and team abbreviation b from matchup column as columns a and b.'''

        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].loc[:, 'a'] = \
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].loc[:, 'matchup'].str.split(' vs. ').str[0].str.strip()

        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].loc[:, 'b'] = \
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].loc[:, 'matchup'].str.split(' vs. ').str[1].str.strip()

        return df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty




    column_name_list_collection_away_home_empty, column_name_list_game_level_season_level = \
    get_column_name_list_and_list_collection_from_team_advanced_box_scores_away_home_team_box_scores_away_home(
        df_team_advanced_box_scores_away_home_team_box_scores_away_home)


    column_name_dict_collection_away_home = \
    get_column_name_dictionary_collection(column_name_list_collection_away_home_empty)


    df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty = \
    get_df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty(
        df_team_advanced_box_scores_away_home_team_box_scores_away_home=df_team_advanced_box_scores_away_home_team_box_scores_away_home,
        column_name_list_collection_away_home_empty=column_name_list_collection_away_home_empty,
        column_name_dict_collection_away_home=column_name_dict_collection_away_home)



    df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty = \
    get_and_add_game_level_season_level_columns_to_df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty_index_2(
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty,
        df_team_advanced_box_scores_away_home_team_box_scores_away_home,
        column_name_list_game_level_season_level)


    df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty = \
    extract_and_add_columns_team_abbreviation_a_and_team_abbreviation_b_as_a_and_b(
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty)





    #get column name lists and dictionary for calculating 12 game moving average for Team Advanced Box Score Stats and Team Box Score Stats.
    column_name_list_numeric = \
    df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2]\
    .select_dtypes('number').drop(columns=['game_id' , 'season_year']).columns.to_list()

    column_name_list_numeric_season = \
    ['season'] + column_name_list_numeric

    column_name_list_numeric_not_team_id = \
    [k for k in column_name_list_numeric if not k in ['team_id']]

    column_name_list_cma12 = \
    [k + '_cma12' for k in column_name_list_numeric_not_team_id]

    column_name_dict_cma12 = \
    dict(zip(column_name_list_numeric_not_team_id, column_name_list_cma12))


    def calculate_12_game_moving_average_for_team_advanced_box_scores_and_team_box_scores(
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty,
        column_name_list_numeric_season):
        '''
        Calculate 12 Game Moving Average for Team Advanced Box Score Stats and Team Box Score Stats'''
        df_team_advanced_box_score_cma12_and_team_box_score_cma12 = \
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].loc[:, column_name_list_numeric_season]\
        .groupby(['season', 'team_id']).rolling(12, min_periods=1).mean()\
        .groupby(['season', 'team_id']).shift(periods=1, fill_value=0)

        df_team_advanced_box_score_cma12_and_team_box_score_cma12 = \
        df_team_advanced_box_score_cma12_and_team_box_score_cma12.reset_index().rename(columns={'level_2':'index'})

        #rename team advanced box score and team box score stat columns with appended _cma12
        df_team_advanced_box_score_cma12_and_team_box_score_cma12 = \
        df_team_advanced_box_score_cma12_and_team_box_score_cma12.rename(columns=column_name_dict_cma12)

        return df_team_advanced_box_score_cma12_and_team_box_score_cma12

    df_team_advanced_box_score_cma12_and_team_box_score_cma12 = \
    calculate_12_game_moving_average_for_team_advanced_box_scores_and_team_box_scores(
        df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty,
        column_name_list_numeric_season)


    #add index to df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2]
    df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2] = \
    df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2].reset_index()

    #combine Team Advanced Box Scores and Team Box Scores with Team Advanced Box Score Stats and Team Box Score Stats 12 Game Moving Average
    df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12 = \
    pd.merge(df_team_advanced_box_scores_and_team_box_scores_collection_away_home_empty[2],
             df_team_advanced_box_score_cma12_and_team_box_score_cma12,
             on=['index', 'team_id', 'season'],
             how='inner').drop(columns=['index'])
    
    
    
    
    #build column name list with the game level, the season level, and the team a abbreviation and  team b abbreviation
    column_name_list_game_level_season_level_a_b = \
    column_name_list_game_level_season_level + ['a', 'b']

    
    #build column name list collection and column name dictionary collection for team a and team b selection and rename
    column_name_list_team_level_cma12 = \
    [k for k in df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12 if not k in column_name_list_game_level_season_level_a_b]

    column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b = \
    {0 : [k + '_a' for k in column_name_list_team_level_cma12],
     1 : [k + '_b' for k in column_name_list_team_level_cma12]}
    
    print('column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[0]: \n' + str(column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[0]) + '\n')
    
    print('column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[1]: \n' + str(column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[1]) + '\n')

    column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[2] = \
    column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[0] + column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[1]

    column_name_dict_team_level_collection_a_cma12_a_and_b_cma12_b = {}

    column_name_dict_team_level_collection_a_cma12_a_and_b_cma12_b[0] = \
    dict(zip(column_name_list_team_level_cma12, column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[0]))

    column_name_dict_team_level_collection_a_cma12_a_and_b_cma12_b[1] = \
    dict(zip(column_name_list_team_level_cma12, column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[1]))





    def get_team_a_rows_and_team_b_rows_of_Team_Advanced_Box_Scores_and_Team_Box_Scores_and_rename_their_column_names(
        df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12,
        column_name_dict_team_level_collection_a_cma12_a_and_b_cma12_b):
        '''
        get team a rows and team b rows of Team Advanced Box Scores and Team Box Scores and rename their column names.'''

        df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff = {}

        #get team a rows
        df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[0] = \
        df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12.loc[
            df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12.loc[:, 'TEAM_ABBREVIATION'] == \
            df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12.loc[:, 'a'], :]

        #rename columns for team _a box scores
        df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[0] = \
        df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[0]\
        .rename(columns=column_name_dict_team_level_collection_a_cma12_a_and_b_cma12_b[0])


        #get team b rows
        df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[1] = \
        df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12.loc[
            df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12.loc[:, 'TEAM_ABBREVIATION'] == \
            df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12.loc[:, 'b'], :]

        #rename columns for team _b advanced box scores and team _b box scores
        df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[1] = \
        df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[1]\
        .rename(columns=column_name_dict_team_level_collection_a_cma12_a_and_b_cma12_b[1])

        return df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff

    df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff = \
    get_team_a_rows_and_team_b_rows_of_Team_Advanced_Box_Scores_and_Team_Box_Scores_and_rename_their_column_names(
        df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12,
        column_name_dict_team_level_collection_a_cma12_a_and_b_cma12_b)


    #combine the Team Advanced Box Scores and Team Box Scores, 
    #the Team Advanced Box Score cma12 and Team Box Score cma12, and 
    #Advanced Box Score cma12 difference and Team Box Score cma12 difference
    #of team a and team b

    df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[2] = \
    pd.merge(df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[0],
             df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[1],
             on=column_name_list_game_level_season_level_a_b,
             how='inner')




    #build column name list collection and column name dictionary for 12 Game Moving Average difference of Team Advanced Box Score and Team Box Score   
    column_name_list_team_level_collection_cma12_a_cma12_b_diff = {}

    column_name_list_team_level_collection_cma12_a_cma12_b_diff[0] = \
    [k for k in column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[2] if '_cma12_a' in k]

    column_name_list_team_level_collection_cma12_a_cma12_b_diff[1] = \
    [k for k in column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[2] if '_cma12_b' in k]

    column_name_list_team_level_collection_cma12_a_cma12_b_diff[2] = \
    [k for k in column_name_list_team_level_collection_a_cma12_a_and_b_cma12_b[2] if '_cma12_' in k]

    column_name_list_team_level_collection_cma12_a_cma12_b_diff[3] = \
    [k + '_diff' for k in column_name_list_cma12]

    column_name_dict_team_level_cma12_a_and_diff = \
    dict(zip(column_name_list_team_level_collection_cma12_a_cma12_b_diff[0], column_name_list_team_level_collection_cma12_a_cma12_b_diff[3]))




    #calculate and store difference for Team Advanced Box Scores and Team Box Scores
    df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[2]\
    .loc[:, column_name_list_team_level_collection_cma12_a_cma12_b_diff[3]] = \
    (df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[2]\
    .loc[:, column_name_list_team_level_collection_cma12_a_cma12_b_diff[0]] - \
    df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff[2]\
    .loc[:, column_name_list_team_level_collection_cma12_a_cma12_b_diff[1]].values).rename(columns=column_name_dict_team_level_cma12_a_and_diff)


    return df_team_advanced_box_scores_team_box_scores_team_advanced_box_scores_cma12_team_box_scores_cma12_collection_a_b_diff
    
    
###########################################################################################################################






'''
Validation functions
'''






