---
layout: post
title: Practical SQL use in relational databases
---

In this post, we will learn how to use SQL to query a [plant disease dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) loaded both as [MySQL](https://www.mysql.com/) and [PostgreSQL](https://www.postgresql.org/) databases. We will also compare the capabilities of each database in general and specific to our example task.

## MySQL vs PostgreSQL - General Comparison

**SQL Compliance**
- PostgreSQL is more SQL compliant and supports more advanced SQL features like window functions, CTEs, INTERSECT, etc.
- MySQL has some deviations from SQL standards but has been improving compliance in recent versions.

**Performance**
- MySQL is known to perform better for read-heavy workloads, which is why it's popular for web applications. It's faster for simple read queries.
- PostgreSQL performs better for complex queries, large datasets, and read-write workloads. It has a more sophisticated query optimizer.

**Replication**
- MySQL has built-in master-slave replication and some clustering solutions, but may require 3rd party tools for more advanced setups.
- PostgreSQL has built-in synchronous replication which makes it easier to setup high availability.

**Concurrency**
- PostgreSQL handles concurrency better with its Multi-Version Concurrency Control (MVCC) architecture. Readers don't block writers.
- MySQL traditionally used table-level locking but has improved with row-level locking in recent versions. Still, it doesn't handle high concurrency as well as PostgreSQL.

**Full Text Search**
- PostgreSQL has built-in full text search capabilities.
- MySQL has full text search but it requires special setup and configuration.

**Data Types**
- PostgreSQL supports a wider range of data types including native support for arrays, hstore (key-value pairs), and advanced geometric/spatial types.
- MySQL has a more limited set of data types. It has spatial extensions but not as advanced as PostGIS.

**Extensibility**
- PostgreSQL is highly extensible. You can create custom data types, operators, index types, etc. Popular extensions include PostGIS, pg_trgm, hstore.
- MySQL is less extensible. It allows user-defined functions but not new data types or operators.

**Stored Procedures**
- Both support stored procedures but PostgreSQL allows them to be written in multiple languages (PL/pgSQL, PL/Python, PL/Perl, etc.)
- MySQL stored procedures are written in SQL with some extensions.

**Licensing**
- Both are open source but MySQL has a dual licensing model - GPL open source and commercial. It's owned by Oracle.
- PostgreSQL is fully open source under the permissive PostgreSQL license. Owned by a non-profit.

## Practical implementation - MySQL

We will now look at step-by-step guides (with code examples) of how to initialize and query the [Kaggle New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) as databases in MySQL and PostgreSQL.

Let us start with MySQL.

**Step 1: Download the Dataset**

1. Go and download the dataset on [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). It's around 3 GB in size.

2. Unzip the downloaded file. You should see directories like `train`, `valid` etc. with subdirectories for each class of plant disease.

**Step 2: Examine the Dataset Structure**

1. The dataset has 38 classes of plant diseases across 14 crop species.

2. The directories are structured as `<crop>___<disease>`. For example, `Tomato___Bacterial_spot`.

3. Inside each directory are the individual leaf images for that crop-disease combination.

**Step 3: Set up MySQL Database**

1. Install the MySQL Server if not already done.

2. Create a new database to store the plant disease data:

   ```sql
   CREATE DATABASE plant_diseases;
   USE plant_diseases;
   ```

3. Create a table to store the image paths and labels:

   ```sql
   CREATE TABLE images (
     id INT AUTO_INCREMENT PRIMARY KEY,
     crop VARCHAR(255),
     disease VARCHAR(255),
     image_path VARCHAR(255)
   );
   ```

**Step 4: Load the Dataset into MySQL**

1. Write a Python script to traverse the unzipped dataset directories and insert the image paths and labels into the MySQL table.

2. Use the `os` module to walk through the directories and the `mysql.connector` library to connect to MySQL and execute insert statements.

   ```python
   import os
   import mysql.connector

   mydb = mysql.connector.connect(
     host="localhost",
     user="yourusername",
     password="yourpassword",
     database="plant_diseases"
   )
   mycursor = mydb.cursor()

   dataset_path = '/path/to/unzipped/dataset'

   for root, dirs, files in os.walk(dataset_path):
       for file in files:
           if file.endswith(".jpg"):
               image_path = os.path.join(root, file)
               label = os.path.basename(root)
               crop, disease = label.split('___')
               
               sql = "INSERT INTO images (crop, disease, image_path) VALUES (%s, %s, %s)"
               val = (crop, disease, image_path)
               mycursor.execute(sql, val)

    mydb.commit()
    ```

3. Run this script to populate the `images` table with the dataset.

**Step 5: Analyze the Data**

1. Now you can use SQL queries to analyze the plant disease data. For example:

   ```sql
   -- Count images per crop
   SELECT crop, COUNT(*) AS count 
   FROM images
   GROUP BY crop;
   
   -- Get the distribution of diseases for a specific crop
   SELECT disease, COUNT(*) AS count
   FROM images 
   WHERE crop = 'Tomato'
   GROUP BY disease;
   ```

2. You can also join this table with other tables containing more information about the crops, diseases, treatments etc. for deeper analysis.

That's it! You now have the New Plant Diseases Dataset loaded into MySQL ready for further exploration.

## Practical implementation - PostgreSQL

Let us now learn how to query using PostgreSQL. The dataset download and analysis step remains same as before. Following that:

**Step 1: Set up PostgreSQL Database**

1. Install PostgreSQL if not already done.

2. Create a new database to store the plant disease data:

   ```sql
   CREATE DATABASE plant_diseases;
   \c plant_diseases;
   ```

3. Create a table to store the image paths and labels:

   ```sql
   CREATE TABLE images (
     id SERIAL PRIMARY KEY,
     crop VARCHAR(255),
     disease VARCHAR(255),
     image_path VARCHAR(255)
   );
   ```

   Note the use of `SERIAL` instead of `AUTO_INCREMENT` for the primary key.

**Step 2: Load the Dataset into PostgreSQL**

1. Write a Python script to traverse the dataset directories and insert the data into PostgreSQL. Use the `psycopg2` library to connect to PostgreSQL and execute SQL statements.

   ```python
   import os
   import psycopg2

   conn = psycopg2.connect(
       host="localhost",
       database="plant_diseases",
       user="yourusername",
       password="yourpassword"
   )
   cur = conn.cursor()

   dataset_path = '/path/to/unzipped/dataset'

   for root, dirs, files in os.walk(dataset_path):
       for file in files:
           if file.endswith(".jpg"):
               image_path = os.path.join(root, file)
               label = os.path.basename(root)
               crop, disease = label.split('___')
               
               cur.execute("INSERT INTO images (crop, disease, image_path) VALUES (%s, %s, %s)", 
                           (crop, disease, image_path))

   conn.commit()
   cur.close()
   conn.close()
   ```

   Note the slight differences in connecting to the database and executing the insert statement compared to MySQL.

2. Run this script to populate the `images` table.

**Step 3: Analyze the Data**

1. You can now use SQL to analyze the data in PostgreSQL. For example:

   ```sql
   -- Count images per crop
   SELECT crop, COUNT(*) AS count
   FROM images
   GROUP BY crop;

   -- Get the distribution of diseases for a specific crop  
   SELECT disease, COUNT(*) AS count
   FROM images
   WHERE crop = 'Tomato'  
   GROUP BY disease;
   ```

2. PostgreSQL provides some additional features compared to MySQL, like better support for JSON data types, full text search, and geospatial queries. Depending on your analysis needs, you could leverage these capabilities.

For example, if you have JSON metadata about each image, you could store and query it like:

```sql
ALTER TABLE images ADD COLUMN metadata JSONB;

UPDATE images 
SET metadata = '{"height": 256, "width": 256, "color_space": "RGB"}'
WHERE id = 1;

SELECT * 
FROM images
WHERE metadata->>'color_space' = 'RGB';
```

That covers the process of loading and analyzing the plant disease dataset in PostgreSQL. The steps are quite similar to MySQL with a few syntax differences.

## MySQL vs PostgreSQL - Comparison on Plant Disease dataset

We will now compare MySQL and PostgreSQL for the specific case of the plant disease dataset we loaded:

**Schema Design**  
The relational schema would be very similar in both databases - a table to store image paths and labels with foreign keys to dimension tables for crops and diseases if needed.

PostgreSQL's ability to create custom types could be useful to create a special 'image' type but not strictly necessary. MySQL's lack of transactional DDL is a drawback during initial schema creation and modifications.

**Data Loading**  
Both databases can load the data from CSV files or using programming languages. 

PostgreSQL's COPY command is very efficient for bulk inserts. MySQL's LOAD DATA INFILE is also fast.

For loading images or metadata, PostgreSQL's native JSON support is an advantage. You could store image metadata as a JSON column. With MySQL you'd have to use a TEXT column and handle parsing in application code.

**Querying**  
Most of the queries needed would be simple filters and aggregations which both can handle well.

For more complex queries that categorize images along multiple dimensions or look for patterns, PostgreSQL's window functions and CTEs would be useful. The query optimizer would also likely generate more efficient execution plans.

Full text search on metadata could be done in PostgreSQL without any additional setup.

**Performance**  
For simple read queries like "select all images for crop X", MySQL would likely be faster, especially with indexing.

As queries get more complex with joins, aggregations and analytics, PostgreSQL is likely to perform better, especially with large data volumes.

Write performance is likely to be better with PostgreSQL due to MVCC.

**Geospatial**  
If there is a geospatial component to the data, like field locations, PostgreSQL with the PostGIS extension would provide advanced capabilities for geospatial indexing and queries that MySQL cannot match.

**Machine Learning**  
For advanced analytics and machine learning, the ability to do more in-database with PostgreSQL is an advantage. You can use extensions like MADlib for in-database ML. With MySQL you'd have to extract data to another tool.

**Summary**  
In summary, while both databases could be made to work, PostgreSQL is likely the better choice for this use case due to:

- Ability to handle complex queries and analytics as the application grows
- Native JSON support for unstructured metadata
- Geospatial capabilities with PostGIS
- Ability to do in-database machine learning
- Better handling of concurrency for write workloads

The main advantage of MySQL would be faster performance for simple read queries, but that's likely outweighed by PostgreSQL's other benefits as the application scales and becomes more complex. The strong developer community and momentum behind PostgreSQL is also a factor in its favor.

---
## References

[1] <a id="ref-1"></a> [dolthub.com: Comparing Benchmarks for Postgres, MySQL, and Dolt](https://www.dolthub.com/blog/2023-12-15-benchmarking-postgres-mysql-dolt/)  
[2] <a id="ref-2"></a> [redswitches.com: MySQL vs PostgreSQL: Understanding the Critical Differences](https://www.redswitches.com/blog/mysql-vs-postgresql/)  
[3] <a id="ref-3"></a> [logit.io: PostgreSQL vs MySQL: Use Cases & Key Differences](https://logit.io/blog/post/postgresql-vs-mysql-use-cases-attributes-to-help-you-choose/)  
[4] <a id="ref-4"></a> [reddit.com: Why Do You Choose MySQL Over Postgres?](https://www.reddit.com/r/node/comments/rv6u8u/why_do_you_choose_mysql_over_postgres/)  
[5] <a id="ref-5"></a> [dbvis.com: PostgreSQL vs MySQL: A Comprehensive Comparison](https://www.dbvis.com/thetable/postgresql-vs-mysql/)  
[6] <a id="ref-6"></a> [bytebase.com: Postgres vs MySQL: The Ultimate Comparison](https://www.bytebase.com/blog/postgres-vs-mysql/)  
[7] <a id="ref-7"></a> [aws.amazon.com: The Difference Between MySQL vs PostgreSQL](https://aws.amazon.com/compare/the-difference-between-mysql-vs-postgresql/)  
[8] <a id="ref-8"></a> [phoenixnap.com: Postgres vs MySQL: Which Database to Use and Why](https://phoenixnap.com/kb/postgres-vs-mysql)  
[9] <a id="ref-9"></a> [fivetran.com: PostgreSQL vs MySQL: Which Should You Use?](https://www.fivetran.com/blog/postgresql-vs-mysql)  
[10] <a id="ref-10"></a> [kinsta.com: PostgreSQL vs MySQL: Which Is the Better Database Solution?](https://kinsta.com/blog/postgresql-vs-mysql/)  
[11] <a id="ref-11"></a> [boltic.io: PostgreSQL Performance vs MySQL: A Comparative Analysis](https://www.boltic.io/blog/postgresql-performance-vs-mysql)  
[12] <a id="ref-12"></a> [integrate.io: PostgreSQL vs MySQL: Which One is Better for Your Use Case?](https://www.integrate.io/blog/postgresql-vs-mysql-which-one-is-better-for-your-use-case/)  
[13] <a id="ref-13"></a> [enterprisedb.com: PostgreSQL vs MySQL: A 360-Degree Comparison](https://www.enterprisedb.com/blog/postgresql-vs-mysql-360-degree-comparison-syntax-performance-scalability-and-features)  
[14] <a id="ref-14"></a> [news.ycombinator.com: PostgreSQL vs MySQL: Which One to Choose?](https://news.ycombinator.com/item?id=35599118)  
[15] <a id="ref-15"></a> [reddit.com: In What Circumstances is MySQL Better than PostgreSQL?](https://www.reddit.com/r/PostgreSQL/comments/tldork/in_what_circumstances_is_mysql_better_than/)  
[16] <a id="ref-16"></a> [news.ycombinator.com: Benchmarking Postgres, MySQL, and Dolt](https://news.ycombinator.com/item?id=35906604)  
[17] <a id="ref-17"></a> [news.ycombinator.com: PostgreSQL vs MySQL: Use Cases and Key Differences](https://news.ycombinator.com/item?id=36945115)  
[18] <a id="ref-18"></a> [postgresqltutorial.com: PostgreSQL vs MySQL: A Comprehensive Comparison](https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-vs-mysql/)  
[19] <a id="ref-19"></a> [mdpi.com: DiaMOS Plant: A Dataset for Diagnosis and Monitoring Plant Disease](https://www.mdpi.com/2073-4395/11/11/2107)  
[20] <a id="ref-20"></a> [ncbi.nlm.nih.gov: CCMT: Dataset for Crop Pest and Disease Detection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10285554/)  
[21] <a id="ref-21"></a> [paperswithcode.com: New Plant Diseases Dataset](https://paperswithcode.com/dataset/new-plant-diseases-dataset)  
[22] <a id="ref-22"></a> [arxiv.org: A Survey on Deep Learning Techniques for Plant Disease Detection](https://arxiv.org/html/2312.07905v1)  
[23] <a id="ref-23"></a> [github.com: Plant Disease Detection - GitHub Topics](https://github.com/topics/plant-disease-detection)  
[24] <a id="ref-24"></a> [github.com: Plant Leaf Disease Detection using Deep Learning](https://github.com/mayur7garg/PlantLeafDiseaseDetection)  
[25] <a id="ref-25"></a> [tensorflow.org: PlantVillage Dataset](https://www.tensorflow.org/datasets/catalog/plant_village)  
[26] <a id="ref-26"></a> [researchgate.net: Datasets for Identification and Classification of Plant Leaf Diseases](https://www.researchgate.net/post/Datasets_for_identification_and_classification_of_plant_leaf_diseases)  
[27] <a id="ref-27"></a> [kaggle.com: Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
[28] <a id="ref-28"></a> [kaggle.com: New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
[29] <a id="ref-29"></a> [kaggle.com: New Plant Diseases Dataset - Code](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/code)  
[30] <a id="ref-30"></a> [github.com: Plant Diseases Training Notebook - Greenathon Plant AI](https://github.com/Rishit-dagli/Greenathon-Plant-AI/blob/main/notebooks/plant-diseases-training.ipynb)  
[31] <a id="ref-31"></a> [toolbox.google.com: Diseases Datasets Search (Excluding Kaggle)](https://toolbox.google.com/datasetsearch/search?query=diseases+-site%3Akaggle.com)  


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