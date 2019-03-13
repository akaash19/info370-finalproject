import pandas as pd
import numpy as np
import math

def create_pick_ban_1_team(data, patch_range_low, patch_range_high):
    """
    takes in data and range of patch numbers and returns dataframe 
    with 1 team's picks and bans, and if the team won the match
    """
    data_df = pd.DataFrame(columns=["pick_1", "pick_2", "pick_3", "pick_4", "pick_5",
                                 "ban_1", "ban_2", "ban_3", "ban_4", "ban_5", "ban_6",
                                 "win"])
    for i in data.index:
        patch = data.at[i, 'patch']
        game_mode = data.at[i, 'game_mode']
        if patch >= patch_range_low and patch <= patch_range_high and game_mode == 2: 
            length = len(data.at[i, 'picks_bans'])
            if  length == 22:
                ban1 = data.at[i, 'picks_bans'][0]["hero_id"]
                ban6 = data.at[i, 'picks_bans'][1]["hero_id"]
                ban2 = data.at[i, 'picks_bans'][2]["hero_id"]
                ban7 = data.at[i, 'picks_bans'][3]["hero_id"]
                ban3 = data.at[i, 'picks_bans'][4]["hero_id"]
                ban8 = data.at[i, 'picks_bans'][5]["hero_id"]

                pick1 = data.at[i, 'picks_bans'][6]["hero_id"]
                pick6 = data.at[i, 'picks_bans'][7]["hero_id"]
                pick7 = data.at[i, 'picks_bans'][8]["hero_id"]
                pick2 = data.at[i, 'picks_bans'][9]["hero_id"]

                ban9 = data.at[i, 'picks_bans'][10]["hero_id"]
                ban4 = data.at[i, 'picks_bans'][11]["hero_id"]
                ban10 = data.at[i, 'picks_bans'][12]["hero_id"]
                ban5 = data.at[i, 'picks_bans'][13]["hero_id"]

                pick8 = data.at[i, 'picks_bans'][14]["hero_id"]
                pick3 = data.at[i, 'picks_bans'][15]["hero_id"]
                pick9 = data.at[i, 'picks_bans'][16]["hero_id"]
                pick4 = data.at[i, 'picks_bans'][17]["hero_id"]  

                ban11 = data.at[i, 'picks_bans'][18]["hero_id"]
                ban6 = data.at[i, 'picks_bans'][19]["hero_id"]

                pick5 = data.at[i, 'picks_bans'][20]["hero_id"]
                pick10 = data.at[i, 'picks_bans'][21]["hero_id"] 

                firstWin = True
                secondWin = False

                team1 = pd.Series({"pick_1": pick1, "pick_2": pick2, "pick_3": pick3, "pick_4": pick4, "pick_5": pick5})
                player1_hero = data.at[i, "players"][0]["hero_id"]
                player1_first_pick = False
                player1_radiant = data.at[i, "players"][0]["isRadiant"]

                for hero in team1:
                    if hero == player1_hero:
                        player1_first_pick = True
                        # this is ran when player1's team has first pick and is radiant
                        if player1_radiant:
                            firstWin = data.at[i, 'radiant_win']
                            secondWin = not data.at[i, 'radiant_win']

                if not player1_first_pick:
                    # print(str(data.at[i, "players"][0]["name"]) + " has 2nd pick and is radiant")
                    secondWin = data.at[i, 'radiant_win']
                    firstWin = not data.at[i, 'radiant_win']

                data_df = data_df.append({"pick_1": pick1, 
                                         "pick_2": pick2, 
                                         "pick_3": pick3, 
                                         "pick_4": pick4, 
                                         "pick_5": pick5,
                                         "ban_1": ban1, 
                                         "ban_2": ban2, 
                                         "ban_3": ban3, 
                                         "ban_4": ban4, 
                                         "ban_5": ban5,
                                         "ban_6": ban6,
                                         "win": firstWin}, ignore_index=True)
                data_df = data_df.append({"pick_1": pick6, 
                                         "pick_2": pick7, 
                                         "pick_3": pick8, 
                                         "pick_4": pick9, 
                                         "pick_5": pick10,
                                         "ban_1": ban6, 
                                         "ban_2": ban7, 
                                         "ban_3": ban8, 
                                         "ban_4": ban9, 
                                         "ban_5": ban10,
                                         "ban_6": ban11,
                                         "win": secondWin}, ignore_index=True)
    return data_df

def create_pick_ban_team(data, patch_range_low, patch_range_high):
    """
    takes in data and range of patch numbers and returns dataframe 
    with 1 team's picks and bans, team_id, and if the team won the match
    """
    data_df = pd.DataFrame(columns=["pick_1", "pick_2", "pick_3", "pick_4", "pick_5",
                                 "ban_1", "ban_2", "ban_3", "ban_4", "ban_5", "ban_6",
                                 "win", "team"])
    for i in data.index:
        patch = data.at[i, 'patch']
        game_mode = data.at[i, 'game_mode']
        if patch >= patch_range_low and patch <= patch_range_high and game_mode == 2: 
            length = len(data.at[i, 'picks_bans'])
            if  length == 22:
                ban1 = data.at[i, 'picks_bans'][0]["hero_id"]
                ban6 = data.at[i, 'picks_bans'][1]["hero_id"]
                ban2 = data.at[i, 'picks_bans'][2]["hero_id"]
                ban7 = data.at[i, 'picks_bans'][3]["hero_id"]
                ban3 = data.at[i, 'picks_bans'][4]["hero_id"]
                ban8 = data.at[i, 'picks_bans'][5]["hero_id"]

                pick1 = data.at[i, 'picks_bans'][6]["hero_id"]
                pick6 = data.at[i, 'picks_bans'][7]["hero_id"]
                pick7 = data.at[i, 'picks_bans'][8]["hero_id"]
                pick2 = data.at[i, 'picks_bans'][9]["hero_id"]

                ban9 = data.at[i, 'picks_bans'][10]["hero_id"]
                ban4 = data.at[i, 'picks_bans'][11]["hero_id"]
                ban10 = data.at[i, 'picks_bans'][12]["hero_id"]
                ban5 = data.at[i, 'picks_bans'][13]["hero_id"]

                pick8 = data.at[i, 'picks_bans'][14]["hero_id"]
                pick3 = data.at[i, 'picks_bans'][15]["hero_id"]
                pick9 = data.at[i, 'picks_bans'][16]["hero_id"]
                pick4 = data.at[i, 'picks_bans'][17]["hero_id"]  

                ban11 = data.at[i, 'picks_bans'][18]["hero_id"]
                ban6 = data.at[i, 'picks_bans'][19]["hero_id"]

                pick5 = data.at[i, 'picks_bans'][20]["hero_id"]
                pick10 = data.at[i, 'picks_bans'][21]["hero_id"] 

                firstWin = True
                secondWin = False

                team1 = pd.Series({"pick_1": pick1, "pick_2": pick2, "pick_3": pick3, "pick_4": pick4, "pick_5": pick5})
                player1_hero = data.at[i, "players"][0]["hero_id"]
                player1_first_pick = False
                player1_radiant = data.at[i, "players"][0]["isRadiant"]

                try:
                    team1_id = data.at[i, "radiant_team"]["team_id"]
                except TypeError:
                    continue
                team2_id = data.at[i, "dire_team"]["team_id"]
                
                for hero in team1:
                    if hero == player1_hero:
                        player1_first_pick = True
                        # this is ran when player1's team has first pick and is radiant
                        if player1_radiant:
                            firstWin = data.at[i, 'radiant_win']
                            secondWin = not data.at[i, 'radiant_win']

                if not player1_first_pick:
                    # print(str(data.at[i, "players"][0]["name"]) + " has 2nd pick and is radiant")
                    secondWin = data.at[i, 'radiant_win']
                    firstWin = not data.at[i, 'radiant_win']
                    team1_id = data.at[i, "dire_team"]["team_id"]
                    team2_id = data.at[i, "radiant_team"]["team_id"]
                    
                data_df = data_df.append({"pick_1": pick1, 
                                         "pick_2": pick2, 
                                         "pick_3": pick3, 
                                         "pick_4": pick4, 
                                         "pick_5": pick5,
                                         "ban_1": ban1, 
                                         "ban_2": ban2, 
                                         "ban_3": ban3, 
                                         "ban_4": ban4, 
                                         "ban_5": ban5,
                                         "ban_6": ban6,
                                         "win": firstWin,
                                         "team": team1_id}, ignore_index=True)
                data_df = data_df.append({"pick_1": pick6, 
                                         "pick_2": pick7, 
                                         "pick_3": pick8, 
                                         "pick_4": pick9, 
                                         "pick_5": pick10,
                                         "ban_1": ban6, 
                                         "ban_2": ban7, 
                                         "ban_3": ban8, 
                                         "ban_4": ban9, 
                                         "ban_5": ban10,
                                         "ban_6": ban11,
                                         "win": secondWin,
                                         "team": team2_id}, ignore_index=True)
    return data_df

def create_picks_team(data, patch_range_low, patch_range_high):
    """
    takes in data and range of patch numbers and returns dataframe 
    with 1 team's picks, team_id, and if the team won the match
    """
    data_df = pd.DataFrame(columns=["pick_1", "pick_2", "pick_3", "pick_4", "pick_5",
                                 "win", "team"])
    for i in data.index: 
        patch = data.at[i, 'patch']
        game_mode = data.at[i, 'game_mode']
        if patch >= patch_range_low and patch <= patch_range_high and game_mode == 2: 
            length = len(data.at[i, 'picks_bans'])
            if  length == 22:
                pick1 = data.at[i, "picks_bans"][6]["hero_id"]
                pick6 = data.at[i, "picks_bans"][7]["hero_id"]
                pick7 = data.at[i, "picks_bans"][8]["hero_id"]
                pick2 = data.at[i, "picks_bans"][9]["hero_id"]

                pick8 = data.at[i, "picks_bans"][14]["hero_id"]
                pick3 = data.at[i, "picks_bans"][15]["hero_id"]
                pick9 = data.at[i, "picks_bans"][16]["hero_id"]
                pick4 = data.at[i, "picks_bans"][17]["hero_id"]  

                pick5 = data.at[i, "picks_bans"][20]["hero_id"]
                pick10 = data.at[i, "picks_bans"][21]["hero_id"] 

                firstWin = True
                secondWin = False
                try:
                    team1_id = data.at[i, "radiant_team"]["team_id"]
                except TypeError:
                    continue
                team2_id = data.at[i, "dire_team"]["team_id"]
                
                team1 = pd.Series({"pick_1": pick1, "pick_2": pick2, "pick_3": pick3, "pick_4": pick4, "pick_5": pick5})
                player1_hero = data.at[i, "players"][0]["hero_id"]
                player1_first_pick = False
                player1_radiant = data.at[i, "players"][0]["isRadiant"]

                for hero in team1:
                    if hero == player1_hero:
                        player1_first_pick = True
                        # this is ran when player1's team has first pick and is radiant
                        if player1_radiant:
                            firstWin = data.at[i, 'radiant_win']
                            secondWin = not data.at[i, 'radiant_win']

                if not player1_first_pick:
                    # print(str(data.at[i, "players"][0]["name"]) + " has 2nd pick and is radiant")
                    secondWin = data.at[i, 'radiant_win']
                    firstWin = not data.at[i, 'radiant_win']
                    team1_id = data.at[i, "dire_team"]["team_id"]
                    team2_id = data.at[i, "radiant_team"]["team_id"]

                data_df = data_df.append({"pick_1": pick1, 
                                         "pick_2": pick2, 
                                         "pick_3": pick3, 
                                         "pick_4": pick4, 
                                         "pick_5": pick5,
                                         "win": firstWin,
                                         "team": team1_id}, ignore_index=True)
                data_df = data_df.append({"pick_1": pick6, 
                                         "pick_2": pick7, 
                                         "pick_3": pick8, 
                                         "pick_4": pick9, 
                                         "pick_5": pick10,
                                         "win": secondWin,
                                         "team": team2_id}, ignore_index=True)
    return data_df

def create_picks(data, patch_range_low, patch_range_high):
    """
    takes in data and range of patch numbers and returns dataframe 
    with 1 team's picks, and if the team won the match
    """
    data_df = pd.DataFrame(columns=["pick_1", "pick_2", "pick_3", "pick_4", "pick_5",
                                 "win"])
    for i in data.index: 
        patch = data.at[i, 'patch']
        game_mode = data.at[i, 'game_mode']
        if patch >= patch_range_low and patch <= patch_range_high and game_mode == 2: 
            length = len(data.at[i, 'picks_bans'])
            if  length == 22:
                pick1 = data.at[i, "picks_bans"][6]["hero_id"]
                pick6 = data.at[i, "picks_bans"][7]["hero_id"]
                pick7 = data.at[i, "picks_bans"][8]["hero_id"]
                pick2 = data.at[i, "picks_bans"][9]["hero_id"]

                pick8 = data.at[i, "picks_bans"][14]["hero_id"]
                pick3 = data.at[i, "picks_bans"][15]["hero_id"]
                pick9 = data.at[i, "picks_bans"][16]["hero_id"]
                pick4 = data.at[i, "picks_bans"][17]["hero_id"]  

                pick5 = data.at[i, "picks_bans"][20]["hero_id"]
                pick10 = data.at[i, "picks_bans"][21]["hero_id"] 

                firstWin = True
                secondWin = False

                team1 = pd.Series({"pick_1": pick1, "pick_2": pick2, "pick_3": pick3, "pick_4": pick4, "pick_5": pick5})
                player1_hero = data.at[i, "players"][0]["hero_id"]
                player1_first_pick = False
                player1_radiant = data.at[i, "players"][0]["isRadiant"]

                for hero in team1:
                    if hero == player1_hero:
                        player1_first_pick = True
                        # this is ran when player1's team has first pick and is radiant
                        if player1_radiant:
                            firstWin = data.at[i, 'radiant_win']
                            secondWin = not data.at[i, 'radiant_win']

                if not player1_first_pick:
                    # print(str(data.at[i, "players"][0]["name"]) + " has 2nd pick and is radiant")
                    secondWin = data.at[i, 'radiant_win']
                    firstWin = not data.at[i, 'radiant_win']

                data_df = data_df.append({"pick_1": pick1, 
                                         "pick_2": pick2, 
                                         "pick_3": pick3, 
                                         "pick_4": pick4, 
                                         "pick_5": pick5,
                                         "win": firstWin}, ignore_index=True)
                data_df = data_df.append({"pick_1": pick6, 
                                         "pick_2": pick7, 
                                         "pick_3": pick8, 
                                         "pick_4": pick9, 
                                         "pick_5": pick10,
                                         "win": secondWin}, ignore_index=True)
    return data_df
    
# 120 heroes in the game

def create_vector(data, patch_range_low, patch_range_high):
    """
    takes in data and range of patch numbers and returns 
    1st dataframe with both teams' picks, and bans
    2nd datafram with if the team that had first pick won
    """
    hero_df = pd.DataFrame()
    win_df = pd.DataFrame()
    
    for i in data.index: 
        patch = data.at[i, 'patch']
        game_mode = data.at[i, 'game_mode']
        if patch >= patch_range_low and patch <= patch_range_high and game_mode == 2: 
            length = len(data.at[i, 'picks_bans'])
            if  length == 22:
                ban1 = data.at[i, "picks_bans"][0]["hero_id"]
                ban6 = data.at[i, "picks_bans"][1]["hero_id"]
                ban2 = data.at[i, "picks_bans"][2]["hero_id"]
                ban7 = data.at[i, "picks_bans"][3]["hero_id"]
                ban3 = data.at[i, "picks_bans"][4]["hero_id"]
                ban8 = data.at[i, "picks_bans"][5]["hero_id"]

                pick1 = data.at[i, "picks_bans"][6]["hero_id"]
                pick6 = data.at[i, "picks_bans"][7]["hero_id"]
                pick7 = data.at[i, "picks_bans"][8]["hero_id"]
                pick2 = data.at[i, "picks_bans"][9]["hero_id"]

                ban9 = data.at[i, "picks_bans"][10]["hero_id"]
                ban4 = data.at[i, "picks_bans"][11]["hero_id"]
                ban10 = data.at[i, "picks_bans"][12]["hero_id"]
                ban5 = data.at[i, "picks_bans"][13]["hero_id"]

                pick8 = data.at[i, "picks_bans"][14]["hero_id"]
                pick3 = data.at[i, "picks_bans"][15]["hero_id"]
                pick9 = data.at[i, "picks_bans"][16]["hero_id"]
                pick4 = data.at[i, "picks_bans"][17]["hero_id"]  

                ban11 = data.at[i, "picks_bans"][18]["hero_id"]
                ban6 = data.at[i, "picks_bans"][19]["hero_id"]

                pick5 = data.at[i, "picks_bans"][20]["hero_id"]
                pick10 = data.at[i, "picks_bans"][21]["hero_id"] 

                radiantWin = data.at[i, "radiant_win"]
                
                team1 = pd.Series({"pick_1": pick1, "pick_2": pick2, "pick_3": pick3, "pick_4": pick4, "pick_5": pick5})
                player1_hero = data.at[i, "players"][0]["hero_id"]
                player1_first_pick = False
                player1_radiant = data.at[i, "players"][0]["isRadiant"]
                picks_bans = {}
                
                firstWin = data.at[i, 'radiant_win']
                
                hero_vector = np.zeros((4, 121))
                
                for hero in team1:
                    if hero == player1_hero:
                        player1_first_pick = True
                if not player1_first_pick:
                    firstWin = not data.at[i, 'radiant_win']

                    
                picks_bans = [pick1, pick2, pick3, pick4, pick5,
                              ban1, ban2, ban3, ban4, ban5, ban6,
                              pick6, pick7, pick8, pick9, pick10,
                              ban6, ban7, ban8, ban9,  ban10, ban11] 
                
                pick_b = True
                for v_row in hero_vector:
                    for pb in picks_bans:
                        if pick_b:
                            for i in np.arange(5):
                                v_row[pb] = 1;
                            pick_b = False
                        else:
                            for i in np.arange(6):
                                v_row[pb] = 1;
                            pick_b = True
                
                try:
                    math.isnan(int(firstWin))
                except TypeError:
                    continue
                
                hero_vector = np.concatenate((hero_vector[0], hero_vector[1], hero_vector[2], hero_vector[3]), axis=None)
                hero_df = hero_df.append(pd.Series(hero_vector), ignore_index=True)
                
                win_df = win_df.append(pd.Series(firstWin), ignore_index=True)

    return hero_df, win_df


def create_pick_ban_both_teams(data, patch_range_low, patch_range_high):
    """
    function that takes in data and range of patch number
    and returns dataframe with radiant picks, dire picks, radiant/dire team ids,
    and if radiant won the match    
    """
    data_df = pd.DataFrame(columns=["r_pick_1", "r_pick_2", "r_pick_3", "r_pick_4", "r_pick_5",
                                     "r_ban_1", "r_ban_2", "r_ban_3", "r_ban_4", "r_ban_5", "r_ban_6",
                                     "d_pick_1", "d_pick_2", "d_pick_3", "d_pick_4", "d_pick_5",
                                     "d_ban_1", "d_ban_2", "d_ban_3", "d_ban_4", "d_ban_5", "d_ban_6",
                                     "radiantWin", "radiant", "dire"])
    for i in data.index: 
        patch = data.at[i, 'patch']
        game_mode = data.at[i, 'game_mode']
        if patch >= patch_range_low and patch <= patch_range_high and game_mode == 2: 
            length = len(data.at[i, 'picks_bans'])
            if  length == 22:
                ban1 = data.at[i, "picks_bans"][0]["hero_id"]
                ban6 = data.at[i, "picks_bans"][1]["hero_id"]
                ban2 = data.at[i, "picks_bans"][2]["hero_id"]
                ban7 = data.at[i, "picks_bans"][3]["hero_id"]
                ban3 = data.at[i, "picks_bans"][4]["hero_id"]
                ban8 = data.at[i, "picks_bans"][5]["hero_id"]

                pick1 = data.at[i, "picks_bans"][6]["hero_id"]
                pick6 = data.at[i, "picks_bans"][7]["hero_id"]
                pick7 = data.at[i, "picks_bans"][8]["hero_id"]
                pick2 = data.at[i, "picks_bans"][9]["hero_id"]

                ban9 = data.at[i, "picks_bans"][10]["hero_id"]
                ban4 = data.at[i, "picks_bans"][11]["hero_id"]
                ban10 = data.at[i, "picks_bans"][12]["hero_id"]
                ban5 = data.at[i, "picks_bans"][13]["hero_id"]

                pick8 = data.at[i, "picks_bans"][14]["hero_id"]
                pick3 = data.at[i, "picks_bans"][15]["hero_id"]
                pick9 = data.at[i, "picks_bans"][16]["hero_id"]
                pick4 = data.at[i, "picks_bans"][17]["hero_id"]  

                ban11 = data.at[i, "picks_bans"][18]["hero_id"]
                ban6 = data.at[i, "picks_bans"][19]["hero_id"]

                pick5 = data.at[i, "picks_bans"][20]["hero_id"]
                pick10 = data.at[i, "picks_bans"][21]["hero_id"] 

                radiantWin = data.at[i, "radiant_win"]
                
                try:
                    team1_id = data.at[i, "radiant_team"]["team_id"]
                    team2_id = data.at[i, "dire_team"]["team_id"]
                except TypeError:
                    continue
                team1 = pd.Series({"pick_1": pick1, "pick_2": pick2, "pick_3": pick3, "pick_4": pick4, "pick_5": pick5})
                player1_hero = data.at[i, "players"][0]["hero_id"]
                player1_first_pick = False
                player1_radiant = data.at[i, "players"][0]["isRadiant"]
                picks_bans = {}

                for hero in team1:
                    if hero == player1_hero:
                        player1_first_pick = True
                        # this is ran when player1's team has first pick and is radiant
                        if player1_radiant:
                            # print(str(data.at[i, "players"][0]["name"]) + " has 1st pick and is radiant")
                            picks_bans = {"r_pick_1": pick1, "r_pick_2": pick2, "r_pick_3": pick3, 
                                            "r_pick_4": pick4, "r_pick_5": pick5,
                                            "r_ban_1": ban1, "r_ban_2": ban2, "r_ban_3": ban3, 
                                            "r_ban_4": ban4, "r_ban_5": ban5, "r_ban_6": ban6,
                                            "d_pick_1": pick6, "d_pick_2": pick7, "d_pick_3": pick8, 
                                            "d_pick_4": pick9, "d_pick_5": pick10,
                                            "d_ban_1": ban6, "d_ban_2": ban7, "d_ban_3": ban8, 
                                            "d_ban_4": ban9, "d_ban_5": ban10, "d_ban_6": ban11}

                if not player1_first_pick:
                        # print(str(data.at[i, "players"][0]["name"]) + " has 2nd pick and is radiant")
                        picks_bans = {"r_pick_1": pick6, "r_pick_2": pick7, "r_pick_3": pick8, 
                                        "r_pick_4": pick9, "r_pick_5": pick10,
                                        "r_ban_1": ban6, "r_ban_2": ban7, "r_ban_3": ban8, 
                                        "r_ban_4": ban9, "r_ban_5": ban10, "r_ban_6": ban11,
                                        "d_pick_1": pick1, "d_pick_2": pick2, "d_pick_3": pick3, 
                                        "d_pick_4": pick4, "d_pick_5": pick5,
                                        "d_ban_1": ban1, "d_ban_2": ban2, "d_ban_3": ban3, 
                                        "d_ban_4": ban4, "d_ban_5": ban5, "d_ban_6": ban6}        

                picks_bans.update({"radiantWin": radiantWin,
                                 "radiant": team1_id,
                                 "dire": team2_id})

                row = pd.Series(picks_bans)

                data_df = data_df.append(row, ignore_index=True)
    return data_df
