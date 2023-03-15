from __future__ import unicode_literals
from re import T
from typing import ChainMap
from hazm import *
import math

class IR:

    def __init__(self) -> None:
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.lemmatizer = Lemmatizer()

    def preprocess(self, content, is_query = 0):
        if is_query == 0:
            normalized_content = []
            tokenized_content = []
            for i in range(len(content)):
                normalized_content.append(self.normalizer.normalize(content[i]))

            for i in range(len(normalized_content)):
                tokenized_content.append(word_tokenize(normalized_content[i]))

            for i in range(len(tokenized_content)):
                for j in range(len(tokenized_content[i])):
                    tokenized_content[i][j] = self.stemmer.stem(tokenized_content[i][j])
                    tokenized_content[i][j] = self.lemmatizer.lemmatize(tokenized_content[i][j])
        else:
            normalizer = Normalizer()
            stemmer = Stemmer()
            lemmatizer = Lemmatizer()
            normalized_content = normalizer.normalize(content)
            tokenized_content = word_tokenize(normalized_content)
            for i in range(len(tokenized_content)):
                tokenized_content[i] = stemmer.stem(tokenized_content[i])
                tokenized_content[i] = lemmatizer.lemmatize(tokenized_content[i])
        return tokenized_content
    
    def get_positional_index(self, content):
        preprocessed_data = self.preprocess(content, 0)
        positional_index = {}
        for docID in range(len(preprocessed_data)):
            for i in range(len(preprocessed_data[docID])):
                if preprocessed_data[docID][i] in positional_index:
                    positional_index[preprocessed_data[docID][i]][0] = positional_index[preprocessed_data[docID][i]][0] + 1
                    if docID in positional_index[preprocessed_data[docID][i]][1]:
                        positional_index[preprocessed_data[docID][i]][1][docID].append(i)
                    else:
                        positional_index[preprocessed_data[docID][i]][1][docID] = [i]
                else:
                    positional_index[preprocessed_data[docID][i]] = []
                    positional_index[preprocessed_data[docID][i]].append(1)
                    positional_index[preprocessed_data[docID][i]].append({})
                    positional_index[preprocessed_data[docID][i]][1][docID] = [i]
        return positional_index

    def tf_idf(self, term_frequency, doc_frequency, N):
        tf = 1 + math.log10(term_frequency)
        idf = math.log10(N / doc_frequency)
        tf_idf_weight = tf * idf
        return tf_idf_weight

    def get_weighted_posting_list(self, content):
        positional_index = self.get_positional_index(content)
        self.delete_stop_words(positional_index)
        weighted_posting_list = {}
        doc_lengths = {}
        for term in positional_index.keys():
            weighted_posting_list[term] = []
            weighted_posting_list[term].append(len(positional_index[term][1].keys()))
            weighted_posting_list[term].append({})
            for docID in positional_index[term][1].keys():
                weighted_posting_list[term][1][docID] = self.tf_idf((len(positional_index[term][1][docID])), weighted_posting_list[term][0], len(content))
                if docID in doc_lengths.keys():
                    doc_lengths[docID] += (weighted_posting_list[term][1][docID] * weighted_posting_list[term][1][docID])
                else:
                    doc_lengths[docID] = weighted_posting_list[term][1][docID] * weighted_posting_list[term][1][docID]
        for docID in doc_lengths.keys():
            doc_lengths[docID] = math.sqrt(doc_lengths[docID])
        return [weighted_posting_list, doc_lengths]

    def get_weighted_query(self, weighted_posting_list, N, query):
        preprocessed_query = self.preprocess(query, 1)
        query_posting = {}
        weighted_query = {}
        for term in preprocessed_query:
            if term in weighted_posting_list:
                if term in query_posting.keys():
                    query_posting[term] += 1
                else:
                    query_posting[term] = 1
        for term in query_posting.keys():
            weighted_query[term] = self.tf_idf(query_posting[term], weighted_posting_list[term][0], N)
        return weighted_query

    def cosine_similarity_search(self, N, weighted_posting_list, doc_lengths, champion_list, query, speedup = 0):
        if speedup == 0:
            weighted_query = self.get_weighted_query(weighted_posting_list, N, query)
            similarity_list = []
            for docID in range(N):
                sigma_dot = 0
                sigma_query = 0
                for term in weighted_query.keys():
                    if docID in weighted_posting_list[term][1].keys():
                        sigma_dot += weighted_query[term] * weighted_posting_list[term][1][docID]
                    sigma_query += (weighted_query[term] * weighted_query[term])
                if docID in doc_lengths.keys():
                    similarity_list.append([docID, sigma_dot / (doc_lengths[docID] * math.sqrt(sigma_query))])
            return sorted(similarity_list, key=lambda x: x[1], reverse=True)
        else:
            weighted_query = self.get_weighted_query(champion_list, N, query)
            similarity_list = []
            for docID in range(N):
                sigma_dot = 0
                sigma_query = 0
                for term in weighted_query.keys():
                    if docID in champion_list[term][1].keys():
                        sigma_dot += weighted_query[term] * champion_list[term][1][docID]
                    sigma_query += (weighted_query[term] * weighted_query[term])
                if docID in doc_lengths.keys():
                    similarity_list.append([docID, sigma_dot / (doc_lengths[docID] * math.sqrt(sigma_query))])
            return sorted(similarity_list, key=lambda x: x[1], reverse=True)
        
    def get_champion_list(self, weighted_posting_list):
        champion_list = {}
        for term in weighted_posting_list:
            champion_list[term] = []
            champion_list[term].append(weighted_posting_list[term][0])
            champion_list[term].append({})
            doc_list = sorted(weighted_posting_list[term][1].keys(), key= lambda x: weighted_posting_list[term][1][x], reverse = True)
            k = 20
            if len(doc_list) < k:
                k = len(doc_list)
            for i in range(k):
                champion_list[term][1][doc_list[i]] = weighted_posting_list[term][1][doc_list[i]]
        return champion_list

    def delete_stop_words(self, positional_index):
        terms = positional_index.keys()
        freq_terms = sorted(terms, key= lambda x: positional_index[x][0], reverse= True)
        for i in range(10):
            #print(freq_terms[i])
            positional_index.pop(freq_terms[i], None)
    
    def search(self, query, positional_index):
        preprocessed_query = self.preprocess(query, 1)

        if len(preprocessed_query) > 1:
            for word in preprocessed_query:
                if word not in positional_index:
                    preprocessed_query.remove(word)

            query_substrings = []
            for i in range(len(preprocessed_query)):
                for j in range(len(preprocessed_query) - i):
                    query_substrings.append(preprocessed_query[j:j+i+1])
            query_substrings.reverse()

            priority = 0
            related_content = {}
            for q in query_substrings:
                merge_list = positional_index[q[0]][1]
                for i in range(len(q) - 1):
                    tmp = {}
                    for doc in merge_list.keys():
                        if doc in positional_index[q[i+1]][1].keys():
                            tmp[doc] = list(set([x + 1 for x in positional_index[q[i]][1][doc]]) & set(positional_index[q[i+1]][1][doc]))
                    merge_list = tmp
                related_content[priority] = merge_list
                priority += 1

            return [query_substrings, priority, related_content]
        else:
            related_content = {}
            for i in range(len(positional_index)):
                if preprocessed_query[0] in positional_index:
                     related_content[0] = positional_index[preprocessed_query[0]][1]
            return [[preprocessed_query], 1, related_content]

    def print_result(self, query_results, title):
        for i in range(query_results[1]):
            print('Sub Query: ',' '.join(query_results[0][i]), end= '\n')
            for doc in query_results[2][i].keys():
                print('Title:', title[doc])

    def print_cosine_search_result(self, query_results, title, k= 10):
        if len(query_results) < k:
            k = len(query_results)
        for i in range(k):
            print(f'[{query_results[i][0]}, {query_results[i][1]}]')
            print(f'Title: {title[query_results[i][0]]}')

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