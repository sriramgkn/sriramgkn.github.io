---
layout: post
title: Map of AWS, GCP, and Azure
---

In this post, we will explore three popular cloud service platforms: Amazon Web Services ([AWS](https://aws.amazon.com/)), Google Cloud Platform ([GCP](https://cloud.google.com/products?hl=en)), and Microsoft [Azure](https://azure.microsoft.com/en-ca). We hope to build up this post asynchronously as we scratch more bits of a grand ecosystem.

Cloud computing has revolutionized the way applications are built, deployed, and scaled. AWS, GCP, and Azure offer a wide array of services and tools to cater to various computing needs. Each of the three have unique strengths and differentiators. Our primary interests with using cloud service providers span four domains - i) Compute, ii) Storage, iii) Machine Learning, and iv) Databases. Eventually, we hope to understand all the components involved in a "full" end-to-end ML deployment pipeline used in modern times.

Let us start with an overview of key compute, storage, machine learning, and database services offered by the three platforms:

## Compute

AWS offers [EC2](https://aws.amazon.com/ec2/) for secure and resizable compute capacity in the cloud with a wide selection of instance types. EC2 provides granular control over the underlying infrastructure. They also offer containerization services [ECS](https://aws.amazon.com/ecs/) and [EKS](https://aws.amazon.com/eks/). Serverless computing is offered with [AWS Lambda](https://aws.amazon.com/lambda/).

GCP's primary compute service is [GCE](https://cloud.google.com/products/compute?hl=en) which allows creating VMs with custom CPU and memory configurations. It provides automatic scaling and OS-level changes. They also offer [GKE](https://cloud.google.com/kubernetes-engine?hl=en) for containerization services.

[Azure Compute](https://azure.microsoft.com/en-gb/products/category/compute) offerings include Virtual Machines for Linux and Windows, and [Azure Kubernetes Service](https://azure.microsoft.com/en-gb/products/kubernetes-service). Serverless computing is offered with [Azure Functions](https://azure.microsoft.com/en-gb/products/functions).

## Storage

[Amazon S3](https://aws.amazon.com/s3/) is AWS's scalable object storage service for storing and retrieving any amount of data. It offers industry-leading durability and integrates with many other AWS services. AWS also has [Amazon EBS](https://aws.amazon.com/ebs/) for block-level storage and [Amazon EFS](https://aws.amazon.com/efs/) for file storage.

GCP has [Cloud Storage](https://cloud.google.com/storage?hl=en), a unified object storage service for unstructured data. [Persistent Disk](https://cloud.google.com/persistent-disk?hl=en) provides network-attached block storage.

Azure offers [Blob Storage](https://azure.microsoft.com/en-gb/products/storage/blobs) for unstructured object data, [Disk Storage](https://azure.microsoft.com/en-gb/products/storage/disks) for virtual machine disks, and [Files](https://azure.microsoft.com/en-gb/products/storage/files) for file shares accessible via the SMB protocol.

## Machine Learning

AWS offers [Amazon SageMaker](https://aws.amazon.com/sagemaker/), a fully managed machine learning platform that enables developers to build, train, and deploy ML models. AWS also has purpose-built AI services like [Amazon Rekognition](https://aws.amazon.com/rekognition/) for image and video analysis, and [Amazon Polly](https://aws.amazon.com/polly/) for text-to-speech.

GCP provides [Vertex AI Platform](https://cloud.google.com/vertex-ai?hl=en), a managed service for training and deploying ML models. It has pre-trained APIs for vision, speech, natural language, and translation.

[Azure Machine Learning](https://azure.microsoft.com/en-gb/products/machine-learning) allows data scientists to build, deploy, and manage ML models. [Azure AI Services](https://azure.microsoft.com/en-us/products/ai-services) are pre-built AI capabilities for vision, speech, language, and more.

## Databases

[Amazon RDS](https://aws.amazon.com/rds/) is AWS's managed relational database service supporting multiple engines like PostgreSQL, MySQL, and SQL Server. [Amazon DynamoDB](https://aws.amazon.com/dynamodb/) is a fully managed NoSQL database. Other options include [Amazon DocumentDB](https://aws.amazon.com/documentdb/), [Amazon Neptune](https://aws.amazon.com/neptune/) graph database, and [Amazon QLDB](https://aws.amazon.com/qldb/) ledger database.

GCP offers [Cloud SQL](https://cloud.google.com/sql?hl=en), a fully managed relational database service for MySQL, PostgreSQL, and SQL Server. [Cloud Spanner](https://cloud.google.com/spanner?hl=en) is a horizontally scalable relational database. [Firestore](https://cloud.google.com/firestore?hl=en) and [Bigtable](https://cloud.google.com/bigtable?hl=en) are NoSQL options.

Azure provides [Azure SQL Database](https://azure.microsoft.com/en-gb/products/azure-sql/database), a managed relational database based on SQL Server. [Azure Cosmos DB](https://learn.microsoft.com/en-us/azure/cosmos-db/introduction) is a globally distributed multi-model AI database. [Azure Database for PostgreSQL](https://azure.microsoft.com/en-gb/products/postgresql) and [Azure Database for MySQL](https://azure.microsoft.com/en-gb/products/mysql) are managed PostgreSQL and MySQL offerings.

Overall, the core offerings are comparable across all three providers. AWS tends to have the broadest and most mature set of services, while GCP emphasizes its strength in data analytics and machine learning. Azure provides tight integration with Microsoft's on-premises technologies. The choice often depends on an organization's specific requirements.

---
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
