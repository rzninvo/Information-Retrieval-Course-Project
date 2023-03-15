from __future__ import unicode_literals
from re import T
from typing import ChainMap
import pandas as pd
from hazm import *
import math
from gensim.models import Word2Vec
from numpy.linalg import norm
import multiprocessing
from random import randrange
import copy
import numpy as np
from collections import Counter
import operator

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
    
    def get_w2v_weighted_query(self, w2v_model, positional_index, N, query):
        preprocessed_query = self.preprocess(query, 1)
        query_vector = np.zeros(300)
        weight_sum = 0
        for term in preprocessed_query:
            query_vector += w2v_model.wv[term] * self.tf_idf(1, len(positional_index[term][1].keys()), N)
            weight_sum += self.tf_idf(1, len(positional_index[term][1].keys()), N)
        query_vector = query_vector / weight_sum
        return query_vector

    def get_w2v_weighted_categorized_query(self, w2v_model, positional_index, N, query):
        preprocessed_query = self.preprocess(query, 1)
        cat = preprocessed_query[preprocessed_query.index(":") + 1]
        preprocessed_query.remove("cat")
        preprocessed_query.remove(":")
        preprocessed_query.remove(cat)
        query_vector = np.zeros(300)
        weight_sum = 0
        for term in preprocessed_query:
            query_vector += w2v_model.wv[term] * self.tf_idf(1, len(positional_index[term][1].keys()), N)
            weight_sum += self.tf_idf(1, len(positional_index[term][1].keys()), N)
        query_vector = query_vector / weight_sum
        return [query_vector, cat]

    def get_doc_vector_list(self, content):
        positional_index = self.get_positional_index(content)
        self.delete_stop_words(positional_index)
        doc_vector_list = [{} for i in range(len(content))]
        for term in positional_index.keys():
            for docID in positional_index[term][1].keys():
                doc_vector_list[docID][term] = self.tf_idf((len(positional_index[term][1][docID])), len(positional_index[term][1].keys()), len(content))
        return doc_vector_list
    
    def get_embedded_doc(self, doc_vector_list, w2v_model):
        embedded_docs = []
        for doc in doc_vector_list:
            doc_vec = np.zeros(300)
            weight_sum = 0
            for token, weight in doc.items():
                try:
                    doc_vec += w2v_model.wv[token] * weight
                    weight_sum += weight
                except KeyError:
                    True
            embedded_docs.append(doc_vec / weight_sum)
        return embedded_docs

    def get_cosine_similarity(self, doc1, doc2):
        score = np.dot(doc1, doc2) / (norm(doc1) * norm(doc2))
        return (score + 1) / 2

    def clustering(self, doc_vec_list, centroid_list):
        infunc_clustered_docs = []
        for i in range(5):
            infunc_clustered_docs.append([0])

        for doc_id in range(len(doc_vec_list)):
            best_similarity = 0
            most_similiar_doc_cluster = 0
            for i in range(5):
                current_similarity = self.get_cosine_similarity(doc_vec_list[doc_id], centroid_list[i])
                if current_similarity > best_similarity:
                    best_similarity = current_similarity
                    most_similiar_doc_cluster = i
            infunc_clustered_docs[most_similiar_doc_cluster].append(doc_id)
        for i in range(5):
            infunc_clustered_docs[i].pop(0)
        return infunc_clustered_docs

    def get_clustered_docs(self, w2v_model, content):
        doc_vector_list = self.get_doc_vector_list(content)
        embedded_docs = self.get_embedded_doc(doc_vector_list, w2v_model)
        centroids = []
        for i in range(5):
            centroids.append(embedded_docs[randrange(len(content))])
        for i in range(300):
            clustered_docs = self.clustering(embedded_docs, centroids)
            for j in range(5):
                cluster_vec_sum = embedded_docs[clustered_docs[j][0]]
                for cluster_doc_id in clustered_docs[j][1:]:
                    cluster_vec_sum += embedded_docs[cluster_doc_id]
                centroids[j] = cluster_vec_sum / len(clustered_docs[j])
        return [clustered_docs, centroids, embedded_docs]
    
    def get_labled_version(self, unlabeled_docs, labeled_docs, topics):
        labelized_docs = {"sport": [], "economy": [], "political": [], "culture": [], "health": []}
        for unlabeled_doc in range(len(unlabeled_docs)):
            unlabeled_doc_similarity = {}
            for labeled_doc_id in range(len(labeled_docs)):
                unlabeled_doc_similarity[labeled_doc_id] = self.get_cosine_similarity(labeled_docs[labeled_doc_id], unlabeled_docs[unlabeled_doc])
            d = Counter(unlabeled_doc_similarity)
            KNN = d.most_common(5)
            KNN_topics = {"sport": 0, "economy": 0, "political": 0, "culture": 0, "health": 0}
            for item in KNN:
                try:
                    KNN_topics[topics[item[0]]] += 1
                except KeyError:
                    True
            doc_topic = max(KNN_topics.items(), key=operator.itemgetter(1))[0]
            labelized_docs[doc_topic].append(unlabeled_doc)
        return labelized_docs

    def clustering_similarity_search(self, w2v_model, positional_index, centroids, clustered_docs, embedded_docs, N, query):
        query_vector = self.get_w2v_weighted_query(w2v_model, positional_index, N, query)

        similarities = []
        for i in range(len(centroids)):
            similarities.append([i, self.get_cosine_similarity(query_vector, centroids[i])])
        similarities = sorted(similarities, key=lambda x: (x[1]))

        clustering_search_results = []
        for i in range(2):
            cluster_no = similarities[i][0]
            for doc in range(len(clustered_docs[cluster_no])):
                clustering_search_results.append([clustered_docs[cluster_no][doc], self.get_cosine_similarity(embedded_docs[clustered_docs[cluster_no][doc]], query_vector)])
        clustering_search_results = sorted(clustering_search_results, key=lambda x: (x[1]))
        clustering_search_results.reverse()
        return clustering_search_results

    def category_search(self,  w2v_model, positional_index, labelized_docs, unlabeled_docs, N, query, k = 10):
        query_vector, cat = self.get_w2v_weighted_categorized_query(w2v_model, positional_index, N, query)
        results = {}
        for doc_id in labelized_docs[cat]:
            results[doc_id] = self.get_cosine_similarity(query_vector, unlabeled_docs[doc_id])

        d = Counter(results)
        return d.most_common(k)

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
    
    def print_clustering_search_results(self, query_results, urls, topics, k= 10):
        if len(query_results) < k:
            k = len(query_results)
        for i in range(k):
            print(topics[query_results[i][0]])
            print(urls[query_results[i][0]])
            print(query_results[i][1])
            print()
    def print_category_search_result(self, query_results, urls):
        for result in query_results:
            print(urls[result[0]])    
    
if __name__ == '__main__':
    option = input()
    if option == '1':
        dataset0 = [pd.read_excel('IR00_3_11k News.xlsx'), pd.read_excel('IR00_3_17k News.xlsx'),
                pd.read_excel('IR00_3_20k News.xlsx')]
        dataset = pd.concat(dataset0, ignore_index=True)
        content = dataset['content']
        urls = dataset['url']
        topics = dataset['topic']
        #content = content[:10000]

        w2v_model = Word2Vec.load("w2v_150k_hazm_300_v2.model")
        positional_index = IR().get_positional_index(content)
        clustered_docs, centroids, embedded_docs = IR().get_clustered_docs(w2v_model, content)

        query = input()
        query_results = IR().clustering_similarity_search(w2v_model, positional_index, centroids, clustered_docs, embedded_docs, len(content), query)
        IR().print_clustering_search_results(query_results, urls, topics)
    if option == '2':
        labeled_dataset0 = [pd.read_excel('IR00_3_11k News.xlsx'), pd.read_excel('IR00_3_17k News.xlsx'),
                pd.read_excel('IR00_3_20k News.xlsx')]
        labeled_dataset = pd.concat(labeled_dataset0, ignore_index=True)
        labeled_content = labeled_dataset['content']
        labeled_urls = labeled_dataset['url']
        topics = labeled_dataset['topic']
        labeled_content = labeled_content[:10000]

        unlabeled_dataset = pd.read_excel('IR1_7k_news.xlsx')
        unlabeled_content = unlabeled_dataset['content']
        unlabeled_urls = unlabeled_dataset['url']

        positional_index = IR().get_positional_index(labeled_content)
        w2v_model = Word2Vec.load("w2v_150k_hazm_300_v2.model")
        labeled_doc_vector_list = IR().get_doc_vector_list(labeled_content)
        labeled_embedded_docs = IR().get_embedded_doc(labeled_doc_vector_list, w2v_model)
        unlabeled_doc_vector_list = IR().get_doc_vector_list(unlabeled_content)
        unlabeled_embedded_docs = IR().get_embedded_doc(unlabeled_doc_vector_list, w2v_model)
        labelized_docs = IR().get_labled_version(unlabeled_embedded_docs, labeled_embedded_docs, topics)
        query = input()
        query_results = IR().category_search(w2v_model, positional_index, labelized_docs, unlabeled_embedded_docs, len(labeled_content), query)
        IR().print_category_search_result(query_results, unlabeled_urls)
