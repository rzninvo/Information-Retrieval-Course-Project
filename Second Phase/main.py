import pandas as pd
from IR_phase2 import IR
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    data = pd.read_excel(r'IR.xlsx')
    content = data['content'].tolist()
    titles = data['title'].tolist()
    weighted_posting_list, doclengths = IR().get_weighted_posting_list(content)
    champion_list = IR().get_champion_list(weighted_posting_list)
    print('Enter 1 for normal calculation or 2 for sped up calculation')
    option = input()
    print('Enter query:')
    query = input()
    if option == '1':
        query_result = IR().cosine_similarity_search(len(content), weighted_posting_list, doclengths, champion_list, query)
        IR().print_cosine_search_result(query_result, titles)
    else:
        query_result = IR().cosine_similarity_search(len(content), weighted_posting_list, doclengths, champion_list, query, speedup= 1)
        IR().print_cosine_search_result(query_result, titles)