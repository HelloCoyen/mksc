from math import log
import pandas as pd

def make_card(coefs, woe_result, odds, score, pdo):
    # odds = P_0 / P_1
    b = pdo / log(2)
    a = score - b * log(odds)
    score_card = pd.DataFrame([['base_score', '-', '-', int(a - b * coefs['intercept_'])]],
                              columns=['Variables', 'Bins', 'Woe', 'Score'])
    for v in coefs:
        if v != 'intercept_':
            woe_result[v].insert(loc=0, column='Variables', value=v)
            woe_result[v].rename(columns={v: "Bins", 'woe_i': 'Woe'}, inplace=True)
            woe_result[v]['Score'] = woe_result[v]['Woe'].apply(lambda x: int(-x * b * coefs[v]))
            score_card = pd.concat([score_card, woe_result[v]])
    score_card.to_excel('result/score_card.xlsx', index=False)
    return score_card
