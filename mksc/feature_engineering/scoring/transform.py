
def transform_score(data, score_card):
    base_score = score_card[score_card['Bins'] == '-']['Score'].values[0]
    data['Score'] = base_score
    for i in range(len(data)):
        score_i = base_score
        for k in set(score_card[score_card['Bins'] != '-']['Variables']):
            bin_score = score_card[(score_card['Woe'] == data.iloc[i][k]) & (score_card['Variables'] == k)]['Score']
            score_i += bin_score.values[0]
        data.iloc[i]['Score'] = score_i
    return data
