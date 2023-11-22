from bertopic import BERTopic
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/nomen/Twitter_Data_Analysis/Topic_modeling/discussion_cleaned.csv", encoding='utf-8')
docs = data.Text.tolist()

print(len(docs))
print(docs[0])

# load the model. Take sone mendatory argument
# transformer model to emmbed the document
# Embedding: process of taking the document and converting it to numerical representation. Bert embeds with a sentence transformer.
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")

# take all the docs and embed and predict. Prediction about where they are going to be clustered
topics, probs = topic_model.fit_transform(docs)

# displays the topic number. -1 are the outliers
topic_info = topic_model.get_topic_info()
print(topic_info)

# Get more words for topic 0
topic_0_words = topic_model.get_topic(0)
print(topic_0_words)

# Get representative documents for topic 0
representative_docs_topic_0 = topic_model.get_representative_docs(0)
print(representative_docs_topic_0)

# see the dat with cluster assigned to it. Create a DataFrame with the assigned topics for each document
df = pd.DataFrame({"topic": topics, "document": docs})
print(df)

# visualize, the larger the circle the bigger the cluster
# the closer they are the more semantically similar they are
topic_model.visualize_topics()
plt.show()

topic_model.visualize_barchart()
plt.show()