---
layout: post
title: Understanding API calling
---

In this post, we will see how APIs work. Specifically we'll learn about the REST architectural style, and how REST APIs can be written with GraphQL.

## What is an API?

An API (Application Programming Interface) is a set of rules, protocols, and tools that allow different software applications to communicate and exchange data with each other [[1](#ref-1)] [[3](#ref-3)]. APIs define the types of requests that can be made between applications, how to make those requests, the data formats that should be used, and the conventions to follow [[1](#ref-1)].

Some key points about APIs:

- APIs act as an interface or middleman between different applications [[3](#ref-3)]
- They allow applications to access specific functionality and data from other applications [[3](#ref-3)] 
- APIs simplify application development by enabling integration of existing services and data [[5](#ref-5)]
- They provide a secure way for application owners to share data and functionality with others [[5](#ref-5)]

## Types of APIs

APIs can be categorized in a few different ways [[1](#ref-1)] [[3](#ref-3)]:

1. By intended audience:
- Open/Public APIs: Publicly available with minimal restrictions 
- Partner APIs: Shared with specific business partners
- Internal/Private APIs: Used within an organization 

2. By architecture style/protocol:
- REST APIs: Use HTTP requests to GET, PUT, POST and DELETE data
- SOAP APIs: Use XML for messaging and transfer over SMTP, HTTP, etc.
- RPC APIs: The oldest and simplest; client executes code on the server
- GraphQL, gRPC, WebSocket APIs: Newer styles gaining popularity

3. By purpose:
- Database APIs: Manage and access databases
- Operating System APIs: For OS-level functionality and services
- Web APIs: Allow interaction with web-based apps
- Program APIs: Extend functionality of programming languages

## How APIs Work

The basic flow of an API call looks like this [[1](#ref-1)] [[5](#ref-5)]:

1. An application makes an API request to retrieve data or perform an action
2. The request is processed through the API's endpoint (URI), headers, and body
3. The API makes a call to the external program or web server 
4. The server sends a response back to the API with the requested data
5. The API transfers the data to the initial requesting application

This all happens in the background without any visibility on the user interface.

## API Authentication and Security

Since APIs can expose sensitive data and functionality, authentication is critical to ensure only authorized users can access them [[2](#ref-2)] [[4](#ref-4)]. Common API authentication methods include:

- HTTP Basic Authentication: Sends credentials as user/password pairs
- API Keys: Unique keys issued to each user to track and control usage
- OAuth: Token-based authentication for secure delegated access
- JWT: JSON Web Tokens for stateless authentication

It's important to follow best practices like using HTTPS, access tokens, rate limiting, and input validation to keep APIs secure [[2](#ref-2)] [[4](#ref-4)].

## Benefits of APIs

APIs provide many benefits to developers and organizations [[1](#ref-1)] [[3](#ref-3)] [[5](#ref-5)]:

- Simplify application development through code reuse and integration 
- Enable automation of workflows and processes
- Improve collaboration between internal teams and external partners
- Allow monetization of data and functionality 
- Provide flexibility and agility in developing new applications
- Reduce information silos and disconnected systems

In summary, APIs are an essential part of modern software development, enabling applications to communicate, share data, and extend functionality. Understanding the different types, architectures, authentication methods, and best practices around APIs is key to building robust and secure applications in today's connected world.

---
### References
[1] <a id="ref-1"></a> [axway.com: Types of APIs Different APIs Explained With Concrete Examples for 2024](https://blog.axway.com/learning-center/apis/basics/different-types-apis)  
[2] <a id="ref-2"></a> [postman.com: What Is API Authentication? Benefits, Methods & Best Practices](https://www.postman.com/api-platform/api-authentication/)  
[3] <a id="ref-3"></a> [geeksforgeeks.org: Types of APIs and Applications of API in Real World](https://www.geeksforgeeks.org/application-programming-interfaces-api-and-its-types/)  
[4] <a id="ref-4"></a> [hubspot.com: API Authentication: What It Is, How It Works, Methods, and More](https://blog.hubspot.com/website/api-authentication)  
[5] <a id="ref-5"></a> [ibm.com: Application Programming Interface (API)](https://www.ibm.com/topics/api)  

_Assisted by claude-3-opus on [perplexity.ai](https://perplexity.ai)_

<!-- -------------------------------------------------------------- -->
<!-- 
regex...
\[(\d)\]
to
 [[$1](#ref-$1)]

\[(\d)\] (.*)
to
[$1] <a id="ref-$1"></a> [display text]($2)  

\[(\d\d)\] (.*)
to
[$1] <a id="ref-$1"></a> [display text]($2)  

Citations:
to
---
### References  
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
Cite like this [[1](#ref-1)], and this [[2](#ref-2)]. Use two extra spaces at end of each line for line break
---
### References  
[1] <a id="ref-1"></a> [display text](hyperlink)  
[2] <a id="ref-2"></a> [display text](hyperlink) 
[3] <a id="ref-3"></a> [display text](hyperlink)   
-->
<!-- -------------------------------------------------------------- -->