from bertopic import BERTopic
import pandas as pd

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
topic_model.get_topic_info()

# gets more words
topic_model.get_topic(0)

# like findThoughts
topic_model.get_representative_docs(0)

# see the dat with cluster assigned to it
df = pd.DataFrame({"topic": topics, "document": docs})
print(df)

# visualize, the larger the circle the bigger the cluster
# the closer they are the more semantically similar they are
print(topic_model.visualize_topics())

print(topic_model.visualize_barchart())
