# date - 15/09/2025

"""
Machine Learning Lab: Document Similarity
Measures
14/09/2025
You are given two text documents:
Doc1: Artificial intelligence and machine learning are transforming healthcare by enabling
early diagnosis and personalized treatment.
Doc2: Machine learning techniques are widely applied in healthcare to support early disease
detection, medical imaging, and treatment recommendations.
Question 1: Cosine Similarity (Document Similarity)
Tasks:
1. Preprocess both documents (convert to lowercase, remove punctuation, split into words).
2. Build a vocabulary of all unique words across both documents.
3. Represent each document as a bag-of-words vector (word counts).
4. Write a function cosine_similarity(vec1, vec2) to compute cosine similarity:
Cosine(A, B) = (A · B) / (||A|| * ||B||)
5. Compute and print the cosine similarity between Doc1 and Doc2.
Question 2: Jaccard Similarity (Set-based Similarity)
You are given the same two documents.
Tasks:
1. Convert each document into a set of unique words.
2. Write a function jaccard_similarity(set1, set2) to compute Jaccard similarity:
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
3. Compute and print the Jaccard similarity between the two sets.
"""

import numpy as np
import re       #for working with regular expression

doc1 = "Artificial intelligence and machine learning are transforming healthcare by enabling early diagnosis and personalized treatment."
doc2 = "Machine learning techniques are widely applied in healthcare to support early disease detection, medical imaging, and treatment recommendations."

def preprocess(doc):
    # Convert to lowercase
    doc = doc.lower()   #this converts the doc into lower case
    # Remove punctuation
    # r for taking backslashes literally
    #anything which is not between a-z and whitespace is replaced with '' in doc
    doc = re.sub(r'[^a-z\s]', '', doc)  
    # Split into words
    words = doc.split()     #split to store list of words as token
    return words

tokens1 = preprocess(doc1)      #the returned list of words r kept in token 1
print(f"this is token 1 : {tokens1}")
tokens2 = preprocess(doc2)
# print(f"this is token 2 : {tokens2}")

#steps for building vocab
    #make set of tokens using the set() function
    #take intersection so there is no duplicates 
    #sort them alphabetically and store in any variable
    
    
def cosine_similarity(vec1, vec2):
    vocab = sorted(set(vec1) & set(vec2))   
    vec1_counts = np.array([vec1.count(word) for word in vocab])   
    print(vec1_counts)
    vec2_counts = np.array([vec2.count(word) for word in vocab])
    dot_product = np.dot(vec1_counts, vec2_counts)
    norm1 = np.linalg.norm(vec1_counts)   #squares the value then add them and take their square root 
    # print(f"norm1 : {norm1}")
    norm2 = np.linalg.norm(vec2_counts) 
    
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

#jaccard similarity is ration of common words(intersection) by total words(union)
def jaccard_similarity(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = len(set1 & set2)
    #print(f"intersection of set1 and set2 : {intersection}")  #shows unique words present in set 1 and set 2
    union = len(set1 | set2)
    
    return intersection / union if union else 0.0

cos_sim = cosine_similarity(tokens1, tokens2)   #calls cosine_similarity definition
jacc_sim = jaccard_similarity(tokens1, tokens2)

print(f"Cosine Similarity: {cos_sim:.4f}")
print(f"Jaccard Similarity: {jacc_sim:.4f}")


#task 2 -> symmetry

#reverse then subtract if lesser then epsilon tolerance then its symmetric

para1 = "The oldest classical British and Latin writings had little or no space between words and could be written in boustrophedon (alternating directions). Over time, text direction (left to right) became standardized. Word dividers and terminal punctuation became common."

para2 = "The first way to divide sentences into groups was the original paragraphos, similar to an underscore at the beginning of the new group.[1] The Greek parágraphos evolved into the pilcrow (¶), which in English manuscripts in the Middle Ages can be seen inserted inline between sentences."

tokens3 = preprocess(para1)
# print(f"tokens3 : {tokens3}")
tokens4 = preprocess(para2)
cos_sim34 = cosine_similarity(tokens3,tokens4)
cos_sim43 = cosine_similarity(tokens4,tokens3)

print(f"cos_sim34 : {cos_sim34}")
print(f"cos_sim43 : {cos_sim43}")

if abs(cos_sim34 - cos_sim43) < 1e-9:
    print("Symmetric")
else :
    print("non-symmetric")
    