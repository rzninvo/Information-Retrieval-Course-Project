import pandas as pd
from IR_phase1 import IR
import matplotlib.pyplot as plt
import math

def ir_plot(positional_index):
    freq_terms = sorted(positional_index.keys(), key=lambda x: positional_index[x][0], reverse=True)
    x1 = []
    y1 = []

    for i in range(len(freq_terms)):
        x1.append(math.log(i + 1, 10))
        y1.append(math.log(positional_index[freq_terms[i]][0], 10))

    plt.plot(x1, y1)
    plt.show()

    IR().delete_stop_words(positional_index)

    freq_words = sorted(positional_index.keys(), key=lambda x: positional_index[x][0], reverse=True)

    x2 = []
    y2 = []

    for i in range(len(freq_words)):
        x2.append(math.log(i + 1, 10))
        y2.append(math.log(positional_index[freq_words[i]][0], 10))

    plt.plot(x2, y2)
    plt.show()

if __name__ == '__main__':
    data = pd.read_excel(r'IR.xlsx')
    content = data['content'].tolist()
    titles = data['title'].tolist()
    positional_index = IR().get_positional_index(content)
    query = input()
    query_results = IR().search(query, positional_index)
    IR().print_result(query_results, titles)
    #ir_plot(positional_index)

