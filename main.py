import pandas as pd
from top2vec import Top2Vec

df = pd.read_csv("C:/Users/nomen/OneDrive/Desktop/Bachelorarbeit/Datenanalyse/discussion_cleaned.csv", encoding='utf-8')

# Convert Text to list
docs = df.Text.tolist()
print(docs[0])

# creating the model:  this expects a list with different strings, with each string being the unique document
model = Top2Vec(docs)

# grab the size and quantity of topics
topic_sizes, topic_nums = model.get_topic_sizes()

# return a list of the topics z.B. 705 at index 0. Topic with the highest frequency is at position 0.
print(topic_sizes)

# return the length of every document: last index + 1 = the number of topics
print(topic_nums)

# rank from most common to least common basis. The top 10 topics
topic_words, word_scores, topic_nums = model.get_topics(5)

# iterate ove the three outputs using zip. Zip allows irritation over multiple lists
# return the most frequent words per topic. These are the words that make the cluster unique
for words, scores, num in zip(topic_words, word_scores, topic_nums):
    print(num)
    print(f"Words: {words}")

# grab the documents that correspond to the topic number (its like findThoughts)
documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=3, num_docs=5)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----")
    print(doc)
    print("-----")
    print()

# top 10 similar doc in the cluster, starts in the centroid of the cluster and goes outwards.
# with each doc we have Score:Degree to which it corresponds to the core cluster
# Score: 0,977  = 97 percent accuracy or similarity of cluster
