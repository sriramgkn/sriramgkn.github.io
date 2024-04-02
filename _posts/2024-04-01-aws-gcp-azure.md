---
layout: post
title: Exploring AWS, GCP, and Azure
---

In this post, we will explore three popular cloud service platforms: Amazon Web Services (AWS), Google Cloud Platform (GCP), and Microsoft Azure.

Cloud computing revolutionized the way applications are built, deployed, and scaled. AWS, GCP, and Azure offer a wide array of services and tools to cater to various computing needs. Each of the three have unique strengths and differentiators. Here we will explore each platform's offerings across compute, storage, databases, networking, and machine learning.

Let us start with an overview of key compute, storage, machine learning, and database services offered by the three platforms:

## Compute

AWS offers [Amazon EC2](https://aws.amazon.com/ec2/) for secure, resizable compute capacity in the cloud with a wide selection of instance types. EC2 provides granular control over the underlying infrastructure [[18](#ref-18)]. AWS also has container services like Amazon ECS and EKS, and serverless computing with AWS Lambda [[4](#ref-4)] [[13](#ref-13)].

GCP's primary compute service is [Google Compute Engine](https://cloud.google.com/products/compute?hl=en) which allows creating VMs with custom CPU and memory configurations. It provides automatic scaling and OS-level changes [[5](#ref-5)]. GCP also offers Google Kubernetes Engine for running containerized applications.

[Azure Compute](https://azure.microsoft.com/en-gb/products/category/compute) offerings include Virtual Machines for Linux and Windows, and Azure Kubernetes Service. Azure Functions provides serverless computing capabilities [[20](#ref-20)].

## Storage

[Amazon S3](https://aws.amazon.com/s3/) is AWS's scalable object storage service for storing and retrieving any amount of data. It offers industry-leading durability and integrates with many other AWS services [[1](#ref-1)] [[10](#ref-10)]. AWS also has [Amazon EBS](https://aws.amazon.com/ebs/) for block-level storage and [Amazon EFS](https://aws.amazon.com/efs/) for file storage [[14](#ref-14)].

GCP has [Cloud Storage](https://cloud.google.com/storage?hl=en), a unified object storage service for unstructured data. [Persistent Disk](https://cloud.google.com/persistent-disk?hl=en) provides network-attached block storage [[20](#ref-20)].

Azure offers [Blob Storage](https://azure.microsoft.com/en-gb/products/storage/blobs) for unstructured object data, [Disk Storage](https://azure.microsoft.com/en-gb/products/storage/disks) for virtual machine disks, and [Files](https://azure.microsoft.com/en-gb/products/storage/files) for file shares accessible via the SMB protocol [[20](#ref-20)].

## Machine Learning

AWS offers [Amazon SageMaker](https://aws.amazon.com/sagemaker/), a fully managed machine learning platform that enables developers to build, train, and deploy ML models. AWS also has purpose-built AI services like Amazon Rekognition for image and video analysis, and Amazon Polly for text-to-speech [[2](#ref-2)] [[7](#ref-7)] [[11](#ref-11)].

GCP provides [Vertex AI Platform](https://cloud.google.com/vertex-ai?hl=en), a managed service for training and deploying ML models. It has pre-trained APIs for vision, speech, natural language, and translation [[20](#ref-20)].

[Azure Machine Learning](https://azure.microsoft.com/en-gb/products/machine-learning) allows data scientists to build, deploy, and manage ML models. Azure Cognitive Services are pre-built AI capabilities for vision, speech, language, and decision [[20](#ref-20)].

## Databases

[Amazon RDS](https://aws.amazon.com/rds/) is AWS's managed relational database service supporting multiple engines like PostgreSQL, MySQL, and SQL Server. [Amazon DynamoDB](https://aws.amazon.com/dynamodb/) is a fully managed NoSQL database. Other options include [Amazon DocumentDB](https://aws.amazon.com/documentdb/), [Amazon Neptune](https://aws.amazon.com/neptune/) graph database, and [Amazon QLDB](https://aws.amazon.com/qldb/) ledger database [[3](#ref-3)] [[8](#ref-8)] [[12](#ref-12)] [[15](#ref-15)] [[17](#ref-17)].

GCP offers [Cloud SQL](https://cloud.google.com/sql?hl=en), a fully managed relational database service for MySQL, PostgreSQL, and SQL Server. [Cloud Spanner](https://cloud.google.com/spanner?hl=en) is a horizontally scalable relational database. [Firestore](https://cloud.google.com/firestore?hl=en) and [Bigtable](https://cloud.google.com/bigtable?hl=en) are NoSQL options [[20](#ref-20)].

Azure provides [Azure SQL Database](https://azure.microsoft.com/en-gb/products/azure-sql/database), a managed relational database based on SQL Server. [Azure Cosmos DB](https://learn.microsoft.com/en-us/azure/cosmos-db/introduction) is a globally distributed multi-model AI database. [Azure Database for PostgreSQL](https://azure.microsoft.com/en-gb/products/postgresql) and [Azure Database for MySQL](https://azure.microsoft.com/en-gb/products/mysql) are managed PostgreSQL and MySQL offerings [[20](#ref-20)].

Overall, the core offerings are comparable across all three providers [[20](#ref-20)]. AWS tends to have the broadest and most mature set of services, while GCP emphasizes its strength in data analytics and machine learning. Azure provides tight integration with Microsoft's on-premises technologies. The choice often depends on an organization's specific requirements.

---
## References
[1] <a id="ref-1"></a> [k21academy.com: AWS Storage: Overview, Types & Benefits](https://k21academy.com/amazon-web-services/aws-solutions-architect/aws-storage-overview-types-benefits/)  
[2] <a id="ref-2"></a> [veritis.com: Top 15 Amazon Web Services Machine Learning Tools in the Cloud](https://www.veritis.com/blog/top-15-aws-machine-learning-tools-in-the-cloud/)  
[3] <a id="ref-3"></a> [aws.amazon.com: Decision Guide for Databases on AWS](https://aws.amazon.com/getting-started/decision-guides/databases-on-aws-how-to-choose/)  
[4] <a id="ref-4"></a> [w3schools.com: AWS Compute Services Overview](https://www.w3schools.com/training/aws/aws-compute-services-overview.php)  
[5] <a id="ref-5"></a> [tudip.com: Compute Services on Google Cloud Platform](https://tudip.com/blog-post/compute-services-on-gcp/)  
[6] <a id="ref-6"></a> [aws.amazon.com: AWS Free Tier - Storage](https://aws.amazon.com/free/storage/)  
[7] <a id="ref-7"></a> [aws.amazon.com: Machine Learning on AWS](https://aws.amazon.com/machine-learning/)  
[8] <a id="ref-8"></a> [bluexp.netapp.com: AWS Databases: The Power of Purpose-Built Database Engines](https://bluexp.netapp.com/blog/aws-cvo-blg-aws-databases-the-power-of-purpose-built-database-engines)  
[9] <a id="ref-9"></a> [geeksforgeeks.org: Introduction to AWS Compute Services](https://www.geeksforgeeks.org/introduction-to-aws-compute/)  
[10] <a id="ref-10"></a> [aws.amazon.com: Decision Guide for Storage on AWS](https://aws.amazon.com/getting-started/decision-guides/storage-on-aws-how-to-choose/)  
[11] <a id="ref-11"></a> [aws.amazon.com: AWS Free Tier - Machine Learning](https://aws.amazon.com/free/machine-learning/)  
[12] <a id="ref-12"></a> [geeksforgeeks.org: Types of Databases in AWS](https://www.geeksforgeeks.org/aws-types-of-databases/)  
[13] <a id="ref-13"></a> [aws.amazon.com: AWS Free Tier - Compute](https://aws.amazon.com/free/compute/)  
[14] <a id="ref-14"></a> [digitalcloud.training: AWS Storage Services Overview](https://digitalcloud.training/aws-storage-services/)  
[15] <a id="ref-15"></a> [aws.amazon.com: AWS Database Services](https://aws.amazon.com/products/databases/)  
[16] <a id="ref-16"></a> [aws.amazon.com: AWS Compute Services](https://aws.amazon.com/products/compute/)  
[17] <a id="ref-17"></a> [docs.aws.amazon.com: What is Amazon Relational Database Service (Amazon RDS)?](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html)  
[18] <a id="ref-18"></a> [aws.amazon.com: What is Compute?](https://aws.amazon.com/what-is/compute/)  
[19] <a id="ref-19"></a> [aws.amazon.com: Learn About AWS Database Services](https://aws.amazon.com/products/databases/learn/)  
[20] <a id="ref-20"></a> [tutorialsdojo.com: AWS Cheat Sheets - Compute Services](https://tutorialsdojo.com/aws-cheat-sheets-compute-services/)  

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
