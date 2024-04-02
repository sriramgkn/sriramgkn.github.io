---
layout: post
title: Understanding API calling
---

In this post, we will see how APIs work. We will learn about two distinct architectural styles: REST and GraphQL. We will also see code examples of i) how REST APIs are implemented in Python using the [Django REST Framework](https://www.django-rest-framework.org/) (DRF), and ii) how GraphQL APIs are implemented in Python with the [Graphene](https://graphene-python.org/) library.

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

APIs are an essential part of modern software development, enabling applications to communicate, share data, and extend functionality. Understanding the different types, architectures, authentication methods, and best practices around APIs is key to building robust and secure applications in today's connected world.

REST APIs are the most common architectural style for designing APIs. In Python, REST APIs are implemented using the [DRF](https://www.django-rest-framework.org/) framework mentioned earlier. Before we look at code examples, let us try to understand the REST architectural style in detail:

## Key Characteristics of REST

1. Uniform Interface
REST defines a uniform interface between clients and servers. This simplifies and decouples the architecture, allowing both parts to evolve independently  [[6](#ref-6)]  [[8](#ref-8)]. The uniform interface includes:

- Resource identification in requests (e.g. URIs in web-based REST)
- Resource manipulation through representations (e.g. JSON, XML)
- Self-descriptive messages with metadata about the resource
- Hypermedia as the engine of application state (HATEOAS) - having links to related resources

2. Client-Server Architecture  
REST uses a client-server architecture where clients and servers have clear separation of concerns  [[7](#ref-7)]  [[9](#ref-9)]. Servers expose APIs and resources that clients consume. They evolve independently, with servers not knowing about client implementations and vice versa.

3. Stateless Interactions
Server-side sessions are not required in REST  [[7](#ref-7)]  [[9](#ref-9)]. Each client request contains all the information needed by the server to process it, like authentication tokens. This improves scalability as servers don't have to store client state between requests.

4. Cacheability
Responses from the server can be labeled as cacheable or non-cacheable  [[7](#ref-7)]  [[9](#ref-9)]. Caching can eliminate some client-server interactions, improving scalability and performance. Caches can live on the server or client-side.

5. Layered System
A REST API can be composed of multiple architectural layers  [[8](#ref-8)]  [[9](#ref-9)]. Layers can encapsulate legacy services, enforce security policies, load balance requests, etc. Layers should be invisible to the client.

6. Code on Demand (optional)
REST allows servers to extend client functionality by transferring executable code, like JavaScript  [[8](#ref-8)]. This is an optional constraint and not commonly used.

## Designing RESTful APIs

When designing APIs to conform to the REST architectural style:

- Use HTTP methods explicitly (GET, POST, PUT, DELETE)  [[6](#ref-6)]  [[10](#ref-10)]
- Be stateless and send complete, independent requests  [[6](#ref-6)]  [[10](#ref-10)] 
- Structure the API around resources, which are accessed via URIs  [[6](#ref-6)]  [[10](#ref-10)]
- Use standard, well-defined data formats like JSON or XML  [[10](#ref-10)]
- Leverage HTTP status codes to indicate errors  [[10](#ref-10)]
- Implement authentication via standard means like OAuth or JWT  [[10](#ref-10)]

Some additional design tips:

- Keep base URLs simple and intuitive
- Use nouns for resources, not verbs 
- Model resource hierarchy with URI paths
- Provide filtering, sorting, field selection and paging for collections
- Version your API if you make breaking changes
- Provide good documentation and examples

Essentially, REST provides a set of architectural constraints that, when applied to web services, make them scalable, loosely coupled, simple, and easy to modify and extend. The key is to apply the constraints uniformly and leverage the infrastructure of the web, like HTTP, while clearly separating client and server responsibilities.

Since 2015, a newer style known as GraphQL has gained popularity for its speed and performance benefits. In Python, GraphQL is packaged into the [Graphene](https://graphene-python.org/) library. Before we look at code examples, let us try to understand the philosophy of GraphQL in detail:

## What is GraphQL?

GraphQL is both a query language and a runtime for fulfilling those queries (API calls) developed by Facebook [[11](#ref-11)] [[12](#ref-12)]. It provides a more efficient, powerful and flexible alternative to REST.

The key characteristics of GraphQL are:

- It allows the client to specify exactly what data it needs [[11](#ref-11)]
- It makes it easier to aggregate data from multiple sources [[11](#ref-11)]  
- It uses a type system to describe data [[12](#ref-12)]

With GraphQL, the client sends a query to the API and gets exactly what it needs, nothing more and nothing less. This solves some common problems with REST like over-fetching and under-fetching of data [[11](#ref-11)].

## GraphQL vs REST

While GraphQL can be used as an alternative to REST, it's not necessarily a replacement for REST. Here are the key differences:

1. Data Fetching
- In REST, you typically gather the data by accessing multiple endpoints [[11](#ref-11)] [[14](#ref-14)]. In GraphQL, you'd simply send a single query to the GraphQL server that includes the concrete data requirements. The server then responds with a JSON object where these requirements are fulfilled [[11](#ref-11)].
- GraphQL allows you to retrieve many resources in a single request, while REST requires loading from multiple URLs [[15](#ref-15)].

2. Over/Under Fetching 
- One of the most common problems with REST is that of over- and under-fetching [[11](#ref-11)]. Over-fetching means that a client downloads more information than is actually required in the app. Under-fetching means a specific endpoint doesn't provide enough of the required information [[11](#ref-11)].
- GraphQL solves this by allowing the client to specify exactly what data it needs [[11](#ref-11)] [[14](#ref-14)]. No more over- or under-fetching.

3. Versioning
- With a REST API, you would typically version the API or have multiple endpoints to account for different data needs [[14](#ref-14)]. 
- With GraphQL, there's no need for versioning, as you can add new fields and types to your GraphQL API without impacting existing queries [[14](#ref-14)].

4. Schema and Type System
- GraphQL uses a strong type system to define the capabilities of an API [[12](#ref-12)]. All the types exposed in an API are written down in a schema using the GraphQL Schema Definition Language [[12](#ref-12)].
- REST has no opinion about what format the data should be in [[14](#ref-14)].

5. Architecture
- GraphQL follows a client-driven architecture [[12](#ref-12)], where the client decides what data it needs and in what format. 
- REST follows a server-driven architecture [[12](#ref-12)], where the server determines the data returned.

6. Community and Ecosystem
- REST has a larger community and ecosystem as it has been around for much longer [[12](#ref-12)].
- GraphQL is a growing community and is being rapidly adopted, especially by companies with complex data fetching requirements [[12](#ref-12)] [[13](#ref-13)].

## When to Use GraphQL vs REST

GraphQL is a good choice when:
- Your data is highly interconnected and you have complex data requirements [[13](#ref-13)]
- You want clients to be able to dictate their data requirements [[13](#ref-13)]
- You need high development velocity and flexibility [[13](#ref-13)]

REST is a good choice when:
- You have simple data requirements [[13](#ref-13)]
- You need extensive caching support [[13](#ref-13)]
- You prioritize simplicity and convention over flexibility [[13](#ref-13)]

In summary, GraphQL provides a different and in many cases more efficient approach to developing APIs than REST. It solves many pain points of REST like over/under-fetching and the need for multiple endpoints. However, REST still has its place and is not going away anytime soon. The choice between GraphQL and REST depends on the specific needs of your application.

Let us now look at code examples (with explanations) of how to implement REST APIs using the Django REST Framework (DRF):

## DRF: Setting up a Model

First, let's define a simple model in Django that we want to expose through our API [[16](#ref-16)] [[17](#ref-17)]. In the `models.py` file of your Django app:

```python
from django.db import models

class Student(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    student_id = models.IntegerField()

    def __str__(self):
        return self.first_name
```

This defines a `Student` model with fields for first name, last name, and student ID.

## DRF: Creating a Serializer

Next, we need to create a serializer for our model [[16](#ref-16)] [[17](#ref-17)]. Serializers allow complex data such as querysets and model instances to be converted to native Python datatypes that can then be easily rendered into JSON or other content types [[16](#ref-16)]. In a new file `serializers.py`:

```python
from rest_framework import serializers
from .models import Student

class StudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Student
        fields = ['id', 'first_name', 'last_name', 'student_id']
```

This serializer class defines the fields from the `Student` model that should be included in the serialized representation.

## DRF: Creating Views

Now we can write the views that will handle the API requests [[16](#ref-16)] [[17](#ref-17)]. In your `views.py`:

```python
from rest_framework import viewsets
from .serializers import StudentSerializer
from .models import Student

class StudentViewSet(viewsets.ModelViewSet):
    queryset = Student.objects.all()
    serializer_class = StudentSerializer
```

This view uses DRF's `ModelViewSet` which provides default `create()`, `retrieve()`, `update()`, `partial_update()`, `destroy()` and `list()` actions [[16](#ref-16)]. We specify the `queryset` that should be used (all `Student` objects) and the `serializer_class` to use for serialization.

## DRF: Configuring URLs

Finally, we need to configure the URLs to map to our viewset [[16](#ref-16)] [[17](#ref-17)]. In your `urls.py`:

```python
from django.urls import include, path
from rest_framework import routers
from .views import StudentViewSet

router = routers.DefaultRouter()
router.register(r'students', StudentViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
```

This sets up the `StudentViewSet` to be accessible at the `/students/` endpoint.

## DRF: Testing the API

With this setup, we now have a fully functional API [[17](#ref-17)]. We can test it by starting the Django development server and navigating to `http://localhost:8000/students/` in a web browser. We should see a browsable API interface provided by DRF where we can view existing `Student` objects and create new ones.

We can also interact with the API programmatically. For example, to get a list of all students, we can send a GET request to `http://localhost:8000/students/`. To create a new student, we can send a POST request to the same URL with the student data in JSON format in the request body.

This is just a basic example, but it demonstrates the key components involved in creating a REST API with Django REST Framework: models, serializers, views, and URL configuration. DRF provides many more features and customization options for more advanced use cases.

Let us now look at code examples (with explanations) of how GraphQL APIs are implemented in Python using the Graphene library:

## Graphene: Defining a Schema

The first step is to define a schema for your GraphQL API using Graphene  [[18](#ref-18)]  [[19](#ref-19)]  [[20](#ref-20)]. The schema describes the data types, fields, and relationships in your API. Here's an example:

```python
import graphene

class Book(graphene.ObjectType):
    title = graphene.String()
    author = graphene.String()
    pages = graphene.Int()

class Query(graphene.ObjectType):
    books = graphene.List(Book)

    def resolve_books(self, info):
        return [
            Book(title="To Kill a Mockingbird", author="Harper Lee", pages=281),
            Book(title="1984", author="George Orwell", pages=328),
        ]

schema = graphene.Schema(query=Query)
```

In this example:

- We define a `Book` type with `title`, `author`, and `pages` fields using Graphene's scalar types (`String`, `Int`)  [[18](#ref-1)]  [[20](#ref-3)].
- We define a `Query` type that has a `books` field which returns a list of `Book` objects  [[18](#ref-18)]  [[20](#ref-20)].
- The `resolve_books` method is a resolver that returns the actual data for the `books` field  [[18](#ref-18)]  [[20](#ref-20)].
- Finally, we create a `Schema` instance with our `Query` type  [[18](#ref-18)]  [[20](#ref-20)].

## Graphene: Executing Queries

Once you have a schema, you can execute queries against it  [[18](#ref-18)]  [[19](#ref-19)]  [[20](#ref-20)]. Here's an example:

```python
query = '''
    query {
        books {
            title
            author
        }
    }
'''
result = schema.execute(query)
print(result.data)
```

This will execute the query and print the result:

```json
{
    "books": [
        {
            "title": "To Kill a Mockingbird",
            "author": "Harper Lee"
        },
        {
            "title": "1984", 
            "author": "George Orwell"
        }
    ]
}
```

Note that the query only requests the `title` and `author` fields, so `pages` is not returned.

## Graphene: Mutations

In addition to querying data, you can also modify data through mutations  [[19](#ref-19)]  [[20](#ref-20)]. Here's an example of defining a mutation to create a new book:

```python
class CreateBook(graphene.Mutation):
    class Arguments:
        title = graphene.String()
        author = graphene.String()
        pages = graphene.Int()

    book = graphene.Field(Book)

    def mutate(self, info, title, author, pages):
        book = Book(title=title, author=author, pages=pages)
        return CreateBook(book=book)

class Mutation(graphene.ObjectType):
    create_book = CreateBook.Field()

schema = graphene.Schema(query=Query, mutation=Mutation)
```

And here's how you would execute this mutation:

```python
mutation = '''
    mutation {
        createBook(title: "The Great Gatsby", author: "F. Scott Fitzgerald", pages: 180) {
            book {
                title
                author
                pages
            }
        }
    }
'''
result = schema.execute(mutation)
print(result.data)
```

This will create a new book and return it in the result:

```json
{
    "createBook": {
        "book": {
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald",
            "pages": 180
        }
    }
}
```

## Graphene: Integrating with Flask

To expose your GraphQL API over HTTP, you can integrate it with a web framework like Flask  [[18](#ref-18)]  [[21](#ref-21)]. Here's an example:

```python
from flask import Flask
from graphene_flask import GraphQLView

app = Flask(__name__)
app.add_url_rule('/graphql', view_func=GraphQLView.as_view('graphql', schema=schema, graphiql=True))

if __name__ == '__main__':
    app.run()
```

This will create a `/graphql` endpoint that accepts GraphQL queries and mutations, and also provides the GraphiQL interactive IDE at the same URL  [[18](#ref-18)]  [[21](#ref-21)].

These are just basic examples, but they demonstrate the core concepts of defining schemas, executing queries and mutations, and integrating with a web framework using Graphene and Python. Graphene provides many more features and integrations for building sophisticated GraphQL APIs.

That's it from me! We hope this post serves as a guide to new learners navigating API design.

---
## References
[1] <a id="ref-1"></a> [axway.com: Types of APIs Different APIs Explained With Concrete Examples for 2024](https://blog.axway.com/learning-center/apis/basics/different-types-apis)  
[2] <a id="ref-2"></a> [postman.com: What Is API Authentication? Benefits, Methods & Best Practices](https://www.postman.com/api-platform/api-authentication/)  
[3] <a id="ref-3"></a> [geeksforgeeks.org: Types of APIs and Applications of API in Real World](https://www.geeksforgeeks.org/application-programming-interfaces-api-and-its-types/)  
[4] <a id="ref-4"></a> [hubspot.com: API Authentication: What It Is, How It Works, Methods, and More](https://blog.hubspot.com/website/api-authentication)  
[5] <a id="ref-5"></a> [ibm.com: Application Programming Interface (API)](https://www.ibm.com/topics/api)  
[6] <a id="ref-6"></a> [restfulapi.net: What is REST? REST API Tutorial](https://restfulapi.net)  
[7] <a id="ref-7"></a> [oreilly.com: Building RESTful Web services with Go](https://www.oreilly.com/library/view/building-restful-web/9781788294287/80e6e100-f69f-4b6c-9291-2fe9446b5cf6.xhtml)  
[8] <a id="ref-8"></a> [packtpub.com: Defining REST and its various architectural styles](https://hub.packtpub.com/defining-rest-and-its-various-architectural-styles/)  
[9] <a id="ref-9"></a> [scrapingbee.com: The Six Characteristics of a REST API](https://www.scrapingbee.com/blog/six-characteristics-of-rest-api/)  
[10] <a id="ref-10"></a> [restfulapi.net: REST Architectural Constraints](https://restfulapi.net/rest-architectural-constraints/)  
[11] <a id="ref-11"></a> [kinsta.com: GraphQL vs REST: What's the Difference?](https://kinsta.com/blog/graphql-vs-rest/)  
[12] <a id="ref-12"></a> [guru99.com: GraphQL vs REST: Key Differences](https://www.guru99.com/graphql-vs-rest-apis.html)  
[13] <a id="ref-13"></a> [news.ycombinator.com: GraphQL vs. REST](https://news.ycombinator.com/item?id=37078606)  
[14] <a id="ref-14"></a> [mobilelive.ca: GraphQL vs REST: What You Didn't Know](https://www.mobilelive.ca/blog/graphql-vs-rest-what-you-didnt-know)  
[15] <a id="ref-15"></a> [apollographql.com: GraphQL vs. REST](https://www.apollographql.com/blog/graphql-vs-rest)  
[16] <a id="ref-16"></a> [radixweb.com: How to Create a REST API with Django REST Framework?](https://radixweb.com/blog/create-rest-api-using-django-rest-framework)  
[17] <a id="ref-17"></a> [djangostars.com: Using the Django REST Framework to Develop APIs](https://djangostars.com/blog/rest-apis-django-development/)  
[18] <a id="ref-18"></a> [code.likeagirl.io: Introduction to GraphQL with Python Graphene and GraphQL](https://code.likeagirl.io/introduction-to-graphql-with-python-graphene-and-graphql-a36412250907?gi=6041bd5007f6)  
[19] <a id="ref-19"></a> [jeffersonheard.github.io: GraphQL in Python with Graphene](https://jeffersonheard.github.io/python/graphql/2018/12/08/graphene-python.html)  
[20] <a id="ref-20"></a> [activestate.com: How to Build a GraphQL Server in Python with Graphene](https://www.activestate.com/blog/how-to-build-a-graphql-server-in-python-with-graphene/)  
[21] <a id="ref-21"></a> [apollographql.com: The Complete API Guide](https://www.apollographql.com/blog/complete-api-guide)  

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