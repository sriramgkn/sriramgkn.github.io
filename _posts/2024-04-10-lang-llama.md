---
layout: post
title: Exploring LangChain and LlamaIndex
---

In this post, we will explore [LangChain](https://www.langchain.com/) and [LlamaIndex](https://www.llamaindex.ai/) - two frameworks that help with application development using large language models.

## LangChain

LangChain is an open-source framework designed to simplify the development of applications using large language models (LLMs). At its core, LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications [[1](#ref-1)] [[2](#ref-2)].

Some key concepts in LangChain include:

- Components: Modular building blocks that are easy to use, like LLM wrappers, prompt templates, and vector stores [[2](#ref-2)] [[8](#ref-8)]
- Chains: Combine multiple components together to accomplish a specific task, making it easier to implement complex applications [[2](#ref-2)] [[8](#ref-8)]
- Agents: Allow LLMs to interact with their environment by making decisions about actions to take [[2](#ref-2)] [[8](#ref-8)]

Here's an example of using LangChain to create a simple prompt template and LLM chain:

```python
from langchain import PromptTemplate, OpenAI, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI(temperature=0) 

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is the capital of France?"

print(llm_chain.run(question))
```

This will output a step-by-step answer to the question "What is the capital of France?" using the OpenAI LLM [[11](#ref-11)].

LangChain also provides capabilities for working with documents, including:

- Document loaders to load data into documents 
- Text splitters to chunk long text
- Embeddings and vector stores to store and retrieve relevant documents [[11](#ref-11)]

## LlamaIndex

LlamaIndex (formerly GPT Index) is a project that provides a central interface to connect your LLM's with external data. It provides data structures to make it easier to work with textual data [[3](#ref-3)].

Some key features of LlamaIndex include:

- A suite of in-memory indices over your unstructured and structured data for use with LLMs
- Offers a comprehensive toolset trading off cost and performance
- Provides data connectors to your common data sources and data formats
- Provides indices that can be used for various LLM tasks such as question-answering, summarization, and text generation [[3](#ref-3)]

Here's an example of using LlamaIndex to build a simple question answering system over a set of documents:

```python
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI

# Load documents 
documents = SimpleDirectoryReader('data').load_data()

# Create an index of the documents
index = GPTSimpleVectorIndex(documents)

# Create a question answering system
query_engine = index.as_query_engine()

# Ask a question
response = query_engine.query("What did the author do growing up?")
print(response)
```

This loads a set of documents, creates a vector index, and uses it to answer a question about the author's childhood based on the documents [[3](#ref-3)].

In summary, LangChain and LlamaIndex are two powerful open-source libraries that make it easier to build applications with large language models by providing abstractions and integrations for common tasks and data sources. LangChain focuses more on chaining LLM components together, while LlamaIndex specializes in connecting LLMs with external data through in-memory indices.

---
## References

[1] <a id="ref-1"></a> [techtarget.com: What Is LangChain and How to Use It: A Guide](https://www.techtarget.com/searchenterpriseai/definition/LangChain)  
[2] <a id="ref-2"></a> [langchain.com: Introduction to LangChain](https://js.langchain.com/docs/get_started/introduction)  
[3] <a id="ref-3"></a> [nanonets.com: LangChain: A Complete Guide & Tutorial](https://nanonets.com/blog/langchain/)  
[4] <a id="ref-4"></a> [geeksforgeeks.org: Introduction to LangChain](https://www.geeksforgeeks.org/introduction-to-langchain/)  
[5] <a id="ref-5"></a> [enterprisedna.co: What is LangChain? A Beginner's Guide with Examples](https://blog.enterprisedna.co/what-is-langchain-a-beginners-guide-with-examples/)  
[6] <a id="ref-6"></a> [ibm.com: What is LangChain?](https://www.ibm.com/topics/langchain)  
[7] <a id="ref-7"></a> [pinecone.io: LangChain Intro: What is LangChain?](https://www.pinecone.io/learn/series/langchain/langchain-intro/)  
[8] <a id="ref-8"></a> [github.com: LangChain Examples](https://github.com/alphasecio/langchain-examples)  
[9] <a id="ref-9"></a> [github.com: LangChain Repository](https://github.com/langchain-ai/langchain)  
[10] <a id="ref-10"></a> [langchain.com: LangChain Cookbook](https://python.langchain.com/cookbook/)  
[11] <a id="ref-11"></a> [python-engineer.com: LangChain Crash Course](https://www.python-engineer.com/posts/langchain-crash-course/)  
[12] <a id="ref-12"></a> [langchain.com: Code Writing with LangChain](https://python.langchain.com/docs/expression_language/cookbook/code_writing/)  
[13] <a id="ref-13"></a> [sitepoint.com: LangChain Python: The Complete Guide](https://www.sitepoint.com/langchain-python-complete-guide/)  
[14] <a id="ref-14"></a> [semaphoreci.com: LangChain: A Beginner's Guide](https://semaphoreci.com/blog/langchain)  
[15] <a id="ref-15"></a> [youtube.com: LangChain Crash Course - Build Apps with Language Models](https://www.youtube.com/watch?v=aywZrzNaKjs)  

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
## References  
[1] <a id="ref-1"></a> [display text](hyperlink)  
[2] <a id="ref-2"></a> [display text](hyperlink) 
[3] <a id="ref-3"></a> [display text](hyperlink)  
_Assisted by claude-3-opus on [perplexity.ai](https://perplexity.ai)_ 
-->
<!-- -------------------------------------------------------------- -->