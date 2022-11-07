import csv
import numpy as np

def get_data(entity_head, entity_tail, best_rank_test, best_sort_test, best_sorted_score, mydict):

    print('\n\n', '='*20, '\n\n')
    print('entity head: ', mydict[entity_head], 'entity tail: ', mydict[entity_tail])
    print('best rank: ', best_rank_test+1)
    input_arr = best_sort_test[: best_rank_test + 10]
    top_score = best_sorted_score[: best_rank_test + 10]
    #print(best_rank_test, np.shape(top_score), np.shape(input_arr))
    print('ordered predicted tails')
    if top_score[0] > 0.3:
        for i in range(len(input_arr)):
            if i == best_rank_test:
                print('(',i+1, ') ', mydict[input_arr[i]], ', probability score=', top_score[i], '----------->')
            else:
                print('(',i+1, ') ', mydict[input_arr[i]], 'score=', top_score[i])

















'''mydict = {}
    with open('entity_to_id_dict.csv', mode='r') as infile:
        reader = csv.reader(infile)
        mydict = dict((int(rows[1]),rows[0]) for rows in reader) #key is id now, then str is val'''

'''
Relation: hasAttackLocation [rdim 200]

entity head:  Attacker_China entity tail:  Country_UK
ordered predicted tails
1523 :  Malware_Maudi
1524 :  Malware_MiniDuke
277 :  Country_UK
1545 :  Malware_Roaming_tiger
'''
'''
Relation: hasAttackLocation [rdim 30]
entity head:  Campaign_ejun0708 entity tail:  Country_Kazakhstan
ordered predicted tails
1509 :  Malware_IRC
1524 :  Malware_MiniDuke
1523 :  Malware_Maudi
1545 :  Malware_Roaming_tiger
1448 :  Malware_57_03_APT1_samples
256 :  Country_Kazakhstan
1536 :  Malware_Poison_Ivy
1564 :  Malware_Stuxnet
1653 :  Organization_CIRCL
1493 :  Malware_Gh0stRat
387 :  ExploitTargetObject_Google
1525 :  Malware_Miniduke
1600 :  Malware_install.exe
1573 :  Malware_Terminator
1505 :  Malware_Hydraq

'''

'''
involvesMalware
entity head:  campaign_operation_aurora 
entity tail:  malware_hydraq

best rank:  9
ordered predicted tails
1203 :  malware_miniduke
1201 :  malware_maudi
1255 :  malware_stuxnet
1190 :  malware_irc
1116 :  malware_57_03_apt1_samples
1222 :  malware_poison_ivy
456 :  exploittargetobject_google
1172 :  malware_gh0strat
1234 :  malware_roaming_tiger
1184 :  malware_hydraq ----------->
1188 :  malware_install.exe
1197 :  malware_keyboy
1117 :  malware_57_04_apt1_samples
1629 :  organization_circl
1145 :  malware_comment_crew
1244 :  malware_sayad_client
1985 :  trojanhorse_iexpl0re
1261 :  malware_taidoor
1118 :  malware_57_05_8381_menupass_samples

'''