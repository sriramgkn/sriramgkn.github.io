---
layout: post
title: Understanding non-relational and vector databases
---

In this post, we will try to gain an intuitive understanding of [MongoDB](https://www.mongodb.com/), [Elasticsearch](https://www.elastic.co/), and [Qdrant](https://qdrant.tech/) with examples.

We will start with a technical overview of each database, then differentiate MongoDB and Elasticsearch technically. Finally, we will build an intuition around all three databases simultaneously with a librarian analogy.

**MongoDB**

MongoDB is a popular NoSQL document database that stores data in flexible, JSON-like documents with dynamic schemas. It is designed for scalability, high performance, and ease of use.

**Key Concepts:**
- Document-oriented data model
- Dynamic schema
- Embedded data and arrays
- Horizontal scaling through sharding
- Replication for high availability

**Example Usage:**
```python
import pymongo

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
collection = db["users"]

# Insert a document
user = {"name": "John", "age": 30, "email": "john@example.com"}
result = collection.insert_one(user)

# Query documents
for doc in collection.find({"age": {"$gt": 25}}):
    print(doc)
```

MongoDB excels in handling unstructured or semi-structured data, making it suitable for content management, mobile apps, IoT, and real-time analytics.

**Elasticsearch**

Elasticsearch is a distributed, RESTful search and analytics engine built on Apache Lucene. It provides near real-time search, powerful full-text capabilities, and scalable data analysis.

**Key Concepts:**
- Inverted index for fast search
- Distributed architecture with shards and replicas
- Aggregations and analytics
- RESTful API

**Example Usage:**
```python
from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Index a document
doc = {"name": "Jane", "age": 28, "city": "New York"}
res = es.index(index="people", body=doc)

# Search for documents
query = {"query": {"match": {"city": "New York"}}}
results = es.search(index="people", body=query)
```

Elasticsearch shines in applications that require fast full-text search, log analysis, business analytics, security event monitoring, and geospatial data analysis.

**Qdrant**

Qdrant is an open-source vector database optimized for storing, searching, and managing high-dimensional vector data, making it suitable for machine learning and AI applications.

**Key Concepts:**
- Specialized for vector data
- Fast vector similarity search
- Real-time analytics and queries
- Efficient storage and indexing of high-dimensional data

**Example Usage:**
```python
import qdrant_client

# Connect to Qdrant
client = qdrant_client.QdrantClient(path="localhost:6333")
collection_info = client.recreate_collection(
    collection_name="vectors",
    vectors_config=qdrant_client.models.VectorsConfig(size=512, distance="Cosine"),
)

# Insert vectors
vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
client.upsert(
    collection_name="vectors",
    points=qdrant_client.models.Payload(vectors=vectors),
)

# Search for similar vectors
search_result = client.search(
    collection_name="vectors",
    query_vector=[1, 1, 1],
    limit=2,
)
```

Qdrant is optimized for applications that require vector similarity search, such as recommendation systems, content-based image/video retrieval, fraud detection, and natural language processing.

Overall, MongoDB excels at handling unstructured data with flexible schemas, Elasticsearch provides powerful search and analytics capabilities, and Qdrant is specialized for efficient vector similarity search. The choice depends on the specific requirements around data structure, search needs, scalability, and performance.

## Summary of technical similarities and differences between MongoDB and Elasticsearch

**Similarities:**

1. **NoSQL**: Both Elasticsearch and MongoDB are NoSQL databases that store data in a flexible, schema-less JSON-like document format.
2. **Distributed Architecture**: They are designed to be distributed across multiple nodes/servers for scalability and high availability.
3. **Horizontal Scaling**: Both can scale horizontally by adding more nodes to the cluster as data grows.
4. **Open Source**: Elasticsearch and MongoDB have open-source versions available.

**Differences:**

1. **Primary Purpose**:
   - Elasticsearch is primarily a distributed search and analytics engine built on Apache Lucene. Its strength lies in full-text search, indexing, and data analysis capabilities [[9](#ref-9)] [[11](#ref-11)] [[13](#ref-13)].
   - MongoDB is a general-purpose, document-oriented database designed for storing and retrieving data flexibly and efficiently [[9](#ref-9)] [[11](#ref-11)].

2. **Data Storage**:
   - Elasticsearch uses an inverted index data structure optimized for fast full-text search and analytics [[11](#ref-11)] [[13](#ref-13)].
   - MongoDB stores data in a more traditional database-like structure using collections and documents [[9](#ref-9)] [[11](#ref-11)].

3. **Query Language**:
   - Elasticsearch provides a powerful query DSL (Domain Specific Language) for complex search queries and aggregations [[9](#ref-9)] [[11](#ref-11)].
   - MongoDB uses a rich query language with support for ad-hoc queries, indexing, and aggregations [[9](#ref-9)] [[11](#ref-11)].

4. **Use Cases**:
   - Elasticsearch excels in applications that require full-text search, log analysis, security event monitoring, and real-time analytics [[9](#ref-9)] [[11](#ref-11)] [[13](#ref-13)].
   - MongoDB is well-suited for applications that require flexible data modeling, high scalability, and real-time data access, such as content management, mobile apps, and IoT [[9](#ref-9)] [[11](#ref-11)].

5. **Data Manipulation**:
   - Elasticsearch is optimized for read operations (search and analytics) rather than write operations [[11](#ref-11)] [[13](#ref-13)].
   - MongoDB supports efficient read and write operations, making it suitable for both data storage and retrieval [[9](#ref-9)] [[11](#ref-11)].

In summary, while both Elasticsearch and MongoDB are NoSQL databases with distributed architectures, Elasticsearch specializes in search and analytics, while MongoDB focuses on flexible data storage and retrieval. The choice between them depends on the specific requirements of your application, such as whether you prioritize search capabilities or data modeling and storage [[9](#ref-9)] [[11](#ref-11)] [[13](#ref-13)].

## Librarian analogy to solidify distinctions between all three platforms

Think of each database (MongoDB, Elasticsearch, Qdrant) as a digital library each with a uniquely skilled librarian. Let us understand how each of the three librarians organize their library.

All three databases can store entire books, beyond just titles or specific fields. The difference lies in how they are optimized for searching and retrieving data.

**Storing and Retrieving Data**

In MongoDB, you would store the entire book content as a single document, along with other metadata like title, author, etc. You can retrieve the entire book document based on its unique identifier or other fields.

Example book document in MongoDB:
```json
{
  "_id": ObjectId("609f5a5c8f7f9c2b9c6d5e4f"),
  "title": "To Kill a Mockingbird",
  "author": "Harper Lee",
  "content": "The entire book content goes here..."
}
```

In Elasticsearch, you would also store the entire book content, typically broken down into smaller fields like chapters or pages, along with metadata fields.

Example book document in Elasticsearch:
```json
{
  "_index": "books",
  "_id": "1",
  "_source": {
    "title": "To Kill a Mockingbird",
    "author": "Harper Lee",
    "chapters": [
      {
        "title": "Chapter 1",
        "content": "The content of chapter 1 goes here..."
      },
      {
        "title": "Chapter 2",
        "content": "The content of chapter 2 goes here..."
      }
    ]
  }
}
```

**Searching for Specific Content**

If you want to search for a specific sentence within the book content, both Elasticsearch and MongoDB can do that, but with different approaches and performance characteristics.

In MongoDB, you can use the `$regex` operator to perform a regular expression search on the book content:
```python
sentence = "Shoot all the bluejays you want, if you can hit 'em, but remember it's a sin to kill a mockingbird."
query = {"content": {"$regex": sentence}}
book = db.books.find_one(query)
```

This query will search for the exact sentence within the `content` field of the book documents. If a match is found, it will return the entire book document that contains the sentence.

In Elasticsearch, you can use the `match_phrase` query to search for the specific sentence:
```python
query = {
    "query": {
        "match_phrase": {
            "content": "Shoot all the bluejays you want, if you can hit 'em, but remember it's a sin to kill a mockingbird."
        }
    }
}
results = es.search(index="books", body=query)
```

Elasticsearch will search for the exact sentence within the `content` field of the book documents and return the relevant book documents that contain the sentence, along with metadata like the chapter or page where the sentence appears.

The key difference is that Elasticsearch is optimized for full-text search and can perform this type of search more efficiently than MongoDB, especially on large datasets. It uses an inverted index to quickly find documents that contain the specified terms or phrases.

**Qdrant: The Vector Similarity Specialist**

Qdrant is like a librarian with a special ability to find similar books based on their content. When you add a book to the library, Qdrant converts the book's content into a unique mathematical representation called a vector.

When you want to find books similar to a specific book, you give Qdrant the book's vector. Qdrant then compares this vector with the vectors of all the other books in the library and returns a list of books that are most similar in content. This is called vector similarity search, and it's useful when you want to find books that are conceptually similar, even if they don't contain the exact same words [[23](#ref-23)] [[29](#ref-29)].

In summary:
- Elasticsearch is like a powerful search engine that excels at fast, full-text search based on the content of the books.
- MongoDB is like a flexible document store that can handle books of different structures and allows for complex queries.
- Qdrant is like a vector similarity specialist that finds books with similar content based on their mathematical representations.

Each librarian (database) has its strengths, and the best choice depends on specific needs around fast text search, flexible data storage, or finding similar items based on their content.

We had a basic yet conceptually important doubt at this point, and probed Claude on it. To our pleasant surprise, Claude gave a pretty interesting and nuanced response. Let's check it out.

### Can vector databases be considered the only "truly" schema-less databases? or maybe there is some structure built into the embedding, but post embedding and storage, the retrieval is entirely nonrelational. As a side note, when you say "query by example" for nosql, is the motivation to train a machine learning model? if so, wouldn't it make NoSQL databases essentially vector databases since ML needs vector embeddings?

You raise some great points about the relationship between vector databases and the concept of being "schemaless". Let me share my perspective on this.

While vector databases are often considered schemaless because they don't enforce a rigid, predefined schema on the data, I wouldn't say they are the only "truly" schemaless databases. Other NoSQL databases like document stores (e.g. MongoDB) and key-value stores (e.g. Redis) also provide a high degree of flexibility in terms of the structure of the data they store.

The key difference is that in vector databases, the primary representation of the data is the vector embedding itself. And you're right that there is some inherent structure in these embeddings based on how they were generated (e.g. which language model was used). But the database itself doesn't impose constraints on what those embeddings represent or how they relate to each other. The relationships and similarities emerge from the vector representations themselves.

So in that sense, the retrieval in vector databases is indeed more "non-relational" compared to other NoSQL databases where you still query based on keys, fields, or document structure to some degree. Vector search is entirely based on the mathematical properties of the embeddings.

Regarding your question about "query by example" in NoSQL databases - this can refer to a few different things:

1. In document databases, it often means providing an example document to match against, and the database returns documents that have a similar structure/fields. This is more about structural similarity than semantic similarity.

2. In some NoSQL databases that support machine learning, you can provide an example data point and ask the database to return similar data points based on a trained ML model. This is closer to what vector databases do.

3. And as you noted, if the NoSQL database is actually storing and querying vector embeddings generated by an ML model, then it essentially becomes a vector database, even if it didn't start out that way.

So in summary, while I wouldn't consider vector databases the only truly schemaless option, they do take the concept of non-relational, similarity-based retrieval to the extreme by operating entirely in the vector space. And you're right that as more NoSQL databases incorporate machine learning and vector embeddings, the line between them and purpose-built vector databases starts to blur.

The key is understanding the core retrieval and similarity mechanisms of the database - whether it's based on keys, document structure, vector math, or a combination. Each has its strengths and use cases.

---
## References

[1] <a id="ref-1"></a> [infoworld.com: What is NoSQL? Databases for a Cloud-Scale Future](https://www.infoworld.com/article/3240644/what-is-nosql-databases-for-a-cloud-scale-future.html)  
[2] <a id="ref-2"></a> [dataversity.net: NoSQL Databases: Advantages and Disadvantages](https://www.dataversity.net/nosql-databases-advantages-and-disadvantages/)  
[3] <a id="ref-3"></a> [dataversity.net: A Brief History of Non-Relational Databases](https://www.dataversity.net/a-brief-history-of-non-relational-databases/)  
[4] <a id="ref-4"></a> [stackoverflow.com: What is NoSQL, How Does it Work, and What Benefits Does it Provide?](https://stackoverflow.com/questions/1145726/what-is-nosql-how-does-it-work-and-what-benefits-does-it-provide)  
[5] <a id="ref-5"></a> [kdnuggets.com: NoSQL Know-It-All Compendium](https://www.kdnuggets.com/2021/05/nosql-know-it-all-compendium.html)  
[6] <a id="ref-6"></a> [altexsoft.com: NoSQL Pros and Cons: When to Use a Non-Relational Database](https://www.altexsoft.com/blog/nosql-pros-cons/)  
[7] <a id="ref-7"></a> [kdnuggets.com: NoSQL Explained: Understanding Key-Value Databases](https://www.kdnuggets.com/2021/04/nosql-explained-understanding-key-value-databases.html)  
[8] <a id="ref-8"></a> [mongodb.com: What is NoSQL?](https://www.mongodb.com/nosql-explained)  
[9] <a id="ref-9"></a> [linkedin.com: Elasticsearch vs MongoDB: A Comparative Analysis](https://www.linkedin.com/pulse/elasticsearch-vs-mongodb-manoj-kumar)  
[10] <a id="ref-10"></a> [javatpoint.com: Elasticsearch vs MongoDB: Key Differences](https://www.javatpoint.com/elasticsearch-vs-mongodb)  
[11] <a id="ref-11"></a> [signoz.io: Elasticsearch vs MongoDB: Which One to Choose?](https://signoz.io/blog/elasticsearch-vs-mongodb/)  
[12] <a id="ref-12"></a> [granulate.io: Elasticsearch vs MongoDB: 5 Key Differences & How to Choose](https://granulate.io/blog/elasticsearch-vs-mongodb-5-key-differences-how-to-choose/)  
[13] <a id="ref-13"></a> [logz.io: Elasticsearch vs MongoDB: The Differences Explained](https://logz.io/blog/elasticsearch-vs-mongodb/)  
[14] <a id="ref-14"></a> [bluexp.netapp.com: Elasticsearch vs MongoDB: 6 Key Differences](https://bluexp.netapp.com/blog/cvo-blg-elasticsearch-vs-mongodb-6-key-differences)  
[15] <a id="ref-15"></a> [sprinkledata.com: Elasticsearch vs MongoDB: A Comprehensive Comparison](https://www.sprinkledata.com/blogs/elasticsearch-vs-mongodb-2023-comparison)  
[16] <a id="ref-16"></a> [intellipaat.com: Elasticsearch vs MongoDB: Differences You Should Know](https://intellipaat.com/blog/elasticsearch-vs-mongodb/)  
[17] <a id="ref-17"></a> [mindmajix.com: MongoDB vs Elasticsearch: Key Differences to Know](https://mindmajix.com/mongodb-vs-elasticsearch)  
[18] <a id="ref-18"></a> [mongodb.com: MongoDB Atlas Search vs Elastic Elasticsearch](https://www.mongodb.com/compare/mongodb-atlas-search-vs-elastic-elasticsearch)  
[19] <a id="ref-19"></a> [reddit.com: Elasticsearch vs MongoDB Atlas Search](https://www.reddit.com/r/elasticsearch/comments/xh80fz/elasticsearch_vs_mongodb_atlas_search/)  
[20] <a id="ref-20"></a> [signoz.io: Elasticsearch vs MongoDB: Which One to Choose?](https://signoz.io/blog/elasticsearch-vs-mongodb/)  
[21] <a id="ref-21"></a> [logz.io: Elasticsearch vs MongoDB: The Differences Explained](https://logz.io/blog/elasticsearch-vs-mongodb/)  
[22] <a id="ref-22"></a> [bluexp.netapp.com: Elasticsearch vs MongoDB: 6 Key Differences](https://bluexp.netapp.com/blog/cvo-blg-elasticsearch-vs-mongodb-6-key-differences)  
[23] <a id="ref-23"></a> [zilliz.com: Qdrant vs Elasticsearch: A Comparative Analysis](https://zilliz.com/comparison/qdrant-vs-elastic)  
[24] <a id="ref-24"></a> [zilliz.com: Qdrant vs MongoDB Atlas: A Comparative Analysis](https://zilliz.com/comparison/qdrant-vs-mongodb%20atlas)  
[25] <a id="ref-25"></a> [discuss.elastic.co: Pros and Cons of Using Elasticsearch as a Vector Database](https://discuss.elastic.co/t/pros-and-cons-of-using-elastic-as-a-vector-database/338733)  
[26] <a id="ref-26"></a> [capellasolutions.com: Elasticsearch vs Vector Databases: Decoding the Best Data Management Solution](https://www.capellasolutions.com/blog/elasticsearch-vs-vector-databases-decoding-the-best-data-management-solution)  
[27] <a id="ref-27"></a> [lakefs.io: 12 Vector Databases to Watch in 2023](https://lakefs.io/blog/12-vector-databases-2023/)  
[28] <a id="ref-28"></a> [news.ycombinator.com: Elasticsearch vs. Vector Databases](https://news.ycombinator.com/item?id=37764489)  
[29] <a id="ref-29"></a> [qdrant.tech: Qdrant Benchmarks](https://qdrant.tech/benchmarks/)  
[30] <a id="ref-30"></a> [stackoverflow.com: How to Use Elasticsearch with MongoDB](https://stackoverflow.com/questions/23846971/how-to-use-elasticsearch-with-mongodb)  
[31] <a id="ref-31"></a> [elastic.co: Getting Started with the MongoDB Connector for Elasticsearch](https://www.elastic.co/guide/en/enterprise-search/current/mongodb-start.html)  
[32] <a id="ref-32"></a> [manticoresearch.com: Vector Search in Databases: A Comprehensive Guide](https://manticoresearch.com/blog/vector-search-in-databases/)  
[33] <a id="ref-33"></a> [youtube.com: Elasticsearch vs MongoDB: Which One to Choose?](https://www.youtube.com/watch?v=gQ1WP77CnfY)  
[34] <a id="ref-34"></a> [qdrant.tech: Qdrant Demo](https://qdrant.tech/demo/)  
[35] <a id="ref-35"></a> [youtube.com: Elasticsearch vs MongoDB: A Comparative Analysis](https://www.youtube.com/watch?v=diT_Gl1DBEs)  
[36] <a id="ref-36"></a> [singlestore.com: Choosing a Vector Database for Your Gen AI Stack](https://www.singlestore.com/blog/choosing-a-vector-database-for-your-gen-ai-stack/)  
[37] <a id="ref-37"></a> [db-engines.com: Elasticsearch vs MySQL vs Qdrant](https://db-engines.com/en/system/Elasticsearch%3BMySQL%3BQdrant)  
[38] <a id="ref-38"></a> [db-engines.com: Elasticsearch vs NuoDB vs Qdrant](https://db-engines.com/en/system/Elasticsearch%3BNuoDB%3BQdrant)  
[39] <a id="ref-39"></a> [linkedin.com: The Rising Star of ML Ops: VectorDB and Why They're the SQL & NoSQL of Embedding](https://www.linkedin.com/pulse/rising-star-ml-ops-vectordb-why-theyre-sql-nosql-embedding-mahule)  
[40] <a id="ref-40"></a> [engineeringatscale.substack.com: Vector Databases: Databases for the AI Era](https://engineeringatscale.substack.com/p/vector-databases-databases-for-the)  
[41] <a id="ref-41"></a> [towardsdatascience.com: 8 Examples to Query a NoSQL Database](https://towardsdatascience.com/8-examples-to-query-a-nosql-database-fc3dd1c9a8c)  
[42] <a id="ref-42"></a> [stackoverflow.blog: Do You Need a Specialized Vector Database to Implement Vector Search Well?](https://stackoverflow.blog/2023/09/20/do-you-need-a-specialized-vector-database-to-implement-vector-search-well/)  
[43] <a id="ref-43"></a> [ojs.aaai.org: Efficient Vector Similarity Search in NoSQL Databases](https://ojs.aaai.org/index.php/AAAI/article/download/4991/4864)  
[44] <a id="ref-44"></a> [reddit.com: How to Encode JSON or NoSQL Data into Vector Embeddings](https://www.reddit.com/r/OpenAI/comments/190xxw7/how_do_encode_json_or_nosql_data_into_vector/)  
[45] <a id="ref-45"></a> [nexla.com: What are Vector Databases and Why are They Important for AI?](https://nexla.com/ai-infrastructure/vector-databases/)  
[46] <a id="ref-46"></a> [v7labs.com: What are Vector Databases?](https://www.v7labs.com/blog/vector-databases)  
[47] <a id="ref-47"></a> [mongodb.com: What are Vector Databases?](https://www.mongodb.com/basics/vector-databases)  
[48] <a id="ref-48"></a> [pinecone.io: What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)  
[49] <a id="ref-49"></a> [couchbase.com: What are Vector Embeddings?](https://www.couchbase.com/blog/what-are-vector-embeddings/)  
[50] <a id="ref-50"></a> [eckerson.com: The Why, What, Who and Where of Vector Databases](https://www.eckerson.com/articles/the-why-what-who-and-where-of-vector-databases)  
[51] <a id="ref-51"></a> [redis.com: Schemaless Databases: What They Are and When to Use Them](https://redis.com/blog/schemaless-databases/)  
[52] <a id="ref-52"></a> [oracle.com: Machine Learning - NoSQL Style](https://blogs.oracle.com/ai-and-datascience/post/machine-learning---nosql-style)  
[53] <a id="ref-53"></a> [weaviate.io: What is a Vector Database?](https://weaviate.io/blog/what-is-a-vector-database)  
[54] <a id="ref-54"></a> [mongodb.com: Schemaless Data Models: Advantages and Disadvantages](https://www.mongodb.com/unstructured-data/schemaless)  
[55] <a id="ref-55"></a> [ssrn.com: NoSQL Databases and Scalable Cloud Storage](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3436631)  
[56] <a id="ref-56"></a> [datacamp.com: NoSQL Databases: What Every Data Scientist Needs to Know](https://www.datacamp.com/blog/nosql-databases-what-every-data-scientist-needs-to-know)  
[57] <a id="ref-57"></a> [mongodb.com: What is NoSQL?](https://www.mongodb.com/nosql-explained)  
[58] <a id="ref-58"></a> [youtube.com: Machine Learning meets NoSQL](https://www.youtube.com/watch?v=nyHUKfTMKDQ)  

_Assisted by claude-3-opus on [perplexity.ai](https://perplexity.ai)_

<!-- -------------------------------------------------------------- -->
<!-- 
sequence: renumber, accumulate, format

to increment numbers, use multiple cursors then emmet shortcuts

regex...
\[(\d+)\]
to
 [[$1](#ref-$1)]

regex...
\[(\d+)\] (.*)
to
[$1] <a id="ref-$1"></a> [display text]($2)  

change "Citations:" to "## References"
-->
<!-- 
Include images like this:  
<figure style="text-align: center; width:100%;">
    <img src="{{site.baseurl}}/images/experimenting_files/experimenting_18_1.svg" alt="___" style="max-width:90%; 
    height: auto; margin:3% auto; display:block;">
    <figcaption>___</figcaption>
</figure> 
-->
<!-- 
Include code snippets like this:  
```python 
def square(x):
    return x**2
``` 
-->
<!-- 
Cite like this [[2](#ref-2)], and this [[3](#ref-3)]. Use two extra spaces at end of each line for line break
---
### References  
[1] <a id="ref-1"></a> [display text](hyperlink)  
[2] <a id="ref-2"></a> [display text](hyperlink) 
[3] <a id="ref-3"></a> [display text](hyperlink)   
-->
<!-- -------------------------------------------------------------- -->