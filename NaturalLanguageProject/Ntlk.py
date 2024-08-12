import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
import math
import random

# NLTK stopwords veri kümesini indir
nltk.download('stopwords')

# Belgeleri yükleme ve okuma
def load_documents(path):
    """
    Belirtilen dizindeki belgeleri yükler.
    Belgelerin dosya adlarının ilk kısmını belge kimliği olarak kullanır.
    """
    documents = {}
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), 'r') as file:
                doc_id = int(filename.split('.')[0])
                documents[doc_id] = file.read()
    return documents

# Belge ön işleme
def preprocess_document(text):
    """
    Bir belgeyi ön işler: Küçük harfe çevirir,
    sayıları ve özel karakterleri kaldırır ve kelimelere böler.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    return words

# Sözlük oluşturma
def create_vocabulary(documents):
    """
    Verilen belgelerden bir sözlük oluşturur.
    Kelimeleri köklerine indirger ve stop words kelimeleri çıkarır.
    """
    vocabulary = defaultdict(list)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for doc_id, text in documents.items():
        words = preprocess_document(text)
        for word in words:
            if word not in stop_words:
                stem = stemmer.stem(word)
                if stem not in vocabulary:
                    vocabulary[stem] = [word]
                else:
                    vocabulary[stem].append(word)
    
    return vocabulary

# Ters indeks yapısı oluşturma
def create_inverted_index(documents):
    """
    Verilen belgelerden ters indeks oluşturur.
    """
    inverted_index = defaultdict(lambda: defaultdict(int))

    for doc_id, text in documents.items():
        words = preprocess_document(text)
        for word in words:
            inverted_index[word][doc_id] += 1

    return inverted_index

# Boolean retrieval
def boolean_retrieval(query, inverted_index, operator="AND"):
    """
    Verilen sorguyu ters indeks kullanarak arar.
    AND veya OR operatörüne göre sorguyu değerlendirir.
    """
    query_terms = query.split()
    if operator == "AND":
        result_docs = set(inverted_index[query_terms[0]])
        for term in query_terms[1:]:
            result_docs &= set(inverted_index[term])
    elif operator == "OR":
        result_docs = set()
        for term in query_terms:
            result_docs |= set(inverted_index[term])
    
    return result_docs

# TF-IDF hesaplama
def calculate_tf_idf(documents, inverted_index):
    """
    Belgelerin TF-IDF puanlarını hesaplar.
    """
    tf_idf_scores = defaultdict(dict)
    N = len(documents)
    
    for term, postings in inverted_index.items():
        df = len(postings)
        
        for doc_id, tf in postings.items():
            tf_idf = (1 + math.log(tf)) * math.log(N / df)
            tf_idf_scores[doc_id][term] = tf_idf
    
    return tf_idf_scores

# Değerlendirme (Evaluation) Fonksiyonları
def precision(relevant_retrieved, retrieved):
    return len(relevant_retrieved) / len(retrieved) if retrieved else 0

def recall(relevant_retrieved, relevant):
    return len(relevant_retrieved) / len(relevant) if relevant else 0

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

def average_precision(relevant, retrieved):
    relevant_retrieved = [1 if doc in relevant else 0 for doc in retrieved]
    precisions = [precision(set(relevant_retrieved[:i+1]), retrieved[:i+1]) for i in range(len(relevant_retrieved)) if relevant_retrieved[i]]
    return sum(precisions) / len(precisions) if precisions else 0

# 3.txt dosyasından 20 sorgu seçme
def select_queries(file_path, num_queries=20):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    selected_lines = random.sample(lines, num_queries)
    return [line.strip() for line in selected_lines]

# Sonuçları dosyaya yazma
def write_results_to_file(vocabulary, inverted_index, tf_idf_scores, vocabulary_file, inverted_index_file, boolean_results_file, tf_idf_results_file, evaluation_file):
    """
    Oluşturulan sözlüğü, ters indeksi, Boolean retrieval ve TF-IDF sonuçlarını belirtilen dosyalara yazma.
    """
    with open(vocabulary_file, 'w') as file:
        for stem in sorted(vocabulary.keys()):
            file.write(f"{stem}: {vocabulary[stem]}\n")
    
    with open(inverted_index_file, 'w') as file:
        for term in sorted(inverted_index.keys()):
            postings = inverted_index[term]
            total_occurrences = sum(postings.values())
            file.write(f"{term}: {total_occurrences}\n")
            for doc_id, frequency in postings.items():
                file.write(f"    {doc_id}: {frequency}\n")

    with open(boolean_results_file, 'w') as file:
        queries = select_queries('belgeler/3.txt')
        for query in queries:
            result = boolean_retrieval(query, inverted_index, "AND")
            file.write(f"AND Sorgu Sonucu: {result}\n")
            result = boolean_retrieval(query, inverted_index, "OR")
            file.write(f"OR Sorgu Sonucu: {result}\n")

    with open(tf_idf_results_file, 'w') as file:
        for doc_id, scores in tf_idf_scores.items():
            file.write(f"Belge {doc_id}:\n")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for term, score in sorted_scores:
                file.write(f"\t{term}: {score}\n")

    with open(evaluation_file, 'w') as file:
        queries_evaluation = {
            'query1': (set([1, 2, 3]), set([1, 2])),  # Örnek veriler: ilgili ve retrieve edilen belgeler
            'query2': (set([2, 3, 4]), set([2, 4, 5]))
        }
        
        for query_id, (relevant_docs, retrieved_docs) in queries_evaluation.items():
            relevant_retrieved_docs = relevant_docs.intersection(retrieved_docs)
            precision_val = precision(relevant_retrieved_docs, retrieved_docs)
            recall_val = recall(relevant_retrieved_docs, relevant_docs)
            f1 = f1_score(precision_val, recall_val)
            file.write(f"{query_id} - Precision: {precision_val}, Recall: {recall_val}, F1 Score: {f1}\n")
        
        average_precisions = [average_precision(rel, ret) for rel, ret in queries_evaluation.values()]
        mean_average_precision = sum(average_precisions) / len(average_precisions)
        file.write(f"Mean Average Precision (MAP): {mean_average_precision}\n")

# Ana fonksiyon
if __name__ == "__main__":
    # Belgeleri yükle
    documents_path = 'belgeler/'
    documents = load_documents(documents_path)

    # Sözlüğü oluştur
    vocabulary = create_vocabulary(documents)

    # Ters indeksi oluştur
    inverted_index = create_inverted_index(documents)

    # TF-IDF puanlarını hesapla
    tf_idf_scores = calculate_tf_idf(documents, inverted_index)

    # Dosya adlarını tanımla
    vocabulary_file = 'vocabulary.txt'
    inverted_index_file = 'inverted_index.txt'
    boolean_results_file = 'boolean_results.txt'
    tf_idf_results_file = 'tf_idf_results.txt'
    evaluation_file = 'evaluation_results.txt'  # Değerlendirme sonuçlarının dosya adı
    
    # Sonuçları dosyalara yaz
    write_results_to_file(vocabulary, inverted_index, tf_idf_scores, vocabulary_file, inverted_index_file, boolean_results_file, tf_idf_results_file, evaluation_file)
