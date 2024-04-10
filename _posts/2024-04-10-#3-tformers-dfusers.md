---
layout: post
title: Exploring Transformers and Diffusers
---

In this post, we will explore [Transformers](https://huggingface.co/docs/transformers/index) and [Diffusers](https://huggingface.co/docs/diffusers/en/index) - two popular generative AI libraries by HuggingFace, both based on the transformer architecture in Python.

## HuggingFace Transformers Library

The HuggingFace Transformers library provides state-of-the-art pre-trained models for natural language processing (NLP), computer vision, and audio tasks. It supports popular transformer architectures like BERT, GPT, RoBERTa, ViT, and more.

Key features of the Transformers library include:
- Thousands of pre-trained models that can be used for transfer learning or fine-tuning on downstream tasks
- Interoperability between PyTorch, TensorFlow, and JAX frameworks
- High-level APIs like `pipeline()` for easy inference on common tasks
- Low-level APIs for more flexibility and customization
- Detailed documentation, tutorials, and an active community

### Installation

The Transformers library can be easily installed with pip:

```bash
pip install transformers
```

### Example: Text Classification

Here's an example of using a pre-trained BERT model for text classification with the `pipeline` API:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

result = classifier("I absolutely loved this movie! The acting was superb.")
print(result)
```

Output:
```
[{'label': 'POSITIVE', 'score': 0.9998801946640015}]
```

### Example: Question Answering

Here's an example of using a pre-trained model for question answering:

```python
from transformers import pipeline

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
question = "Where is the Eiffel Tower located?"

result = qa_model(question=question, context=context)
print(result)
```

Output:
```
{'score': 0.9940124392509461, 'start': 35, 'end': 47, 'answer': 'Champ de Mars'}
```

The Transformers library provides a wide range of capabilities for NLP tasks. You can explore more examples and tutorials in the official documentation [[4](#ref-4)].

## HuggingFace Diffusers Library

The HuggingFace Diffusers library focuses on diffusion models for generative tasks like image generation, audio generation, and even generating 3D structures of molecules. It provides pre-trained diffusion models, interchangeable noise schedulers, and modular components for building custom diffusion systems.

Key features of the Diffusers library include:
- State-of-the-art diffusion pipelines for inference with just a few lines of code
- Flexibility to balance trade-offs between generation speed and quality
- Modular design for creating custom end-to-end diffusion systems
- Integration with the Hugging Face Hub for sharing and discovering models

### Installation

The Diffusers library can be easily installed with pip:

```bash
pip install diffusers
```

### Example: Text-to-Image Generation

Here's an example of using a pre-trained Stable Diffusion model for text-to-image generation:

```python
from diffusers import DiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
image.save("astronaut_horse.png")
```

This code snippet loads the Stable Diffusion v1.5 model, moves it to the GPU, and generates an image based on the provided text prompt [[1](#ref-1)] [[3](#ref-3)].

### Example: Image-to-Image Translation

Here's an example of using a pre-trained model for image-to-image translation:

```python
from diffusers import DiffusionPipeline
import requests
from PIL import Image
from io import BytesIO

# Load the image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

model_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "A fantasy landscape, trending on artstation"

images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("fantasy_landscape.png")
```

This code snippet loads an input image, resizes it, and then uses the Stable Diffusion model to generate a new image based on the input image and the provided text prompt [[1](#ref-1)] [[3](#ref-3)].

The Diffusers library provides a powerful toolset for generative tasks using diffusion models. You can explore more examples and tutorials in the official documentation [[1](#ref-1)] [[11](#ref-11)] [[18](#ref-18)].

In summary, the Hugging Face Transformers and Diffusers libraries are invaluable tools for anyone working with state-of-the-art models in NLP, computer vision, and generative AI. They provide pre-trained models, easy-to-use APIs, and extensive documentation to help you get started quickly and build impressive applications [[4](#ref-4)] [[10](#ref-10)] [[12](#ref-12)].

---
## References

Here are the citations with intelligent display text:

[1] <a id="ref-1"></a> [towardsdatascience.com: Hugging Face Just Released the Diffusers Library](https://towardsdatascience.com/hugging-face-just-released-the-diffusers-library-846f32845e65)  
[2] <a id="ref-2"></a> [microsoft.com: What are Hugging Face Transformers? - Azure Databricks](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/train-model/huggingface/)  
[3] <a id="ref-3"></a> [learnopencv.com: Introduction to Hugging Face Diffusers](https://learnopencv.com/hugging-face-diffusers/)  
[4] <a id="ref-4"></a> [freecodecamp.org: Hugging Face Transformer Library Overview](https://www.freecodecamp.org/news/hugging-face-transformer-library-overview/)  
[5] <a id="ref-5"></a> [philschmid.de: Hugging Face Transformers Examples](https://www.philschmid.de/huggingface-transformers-examples)  
[6] <a id="ref-6"></a> [datacamp.com: An Introduction to Using Transformers and Hugging Face](https://www.datacamp.com/tutorial/an-introduction-to-using-transformers-and-hugging-face)  
[7] <a id="ref-7"></a> [youtube.com: Hugging Face Transformers Tutorial - Getting Started with NLP](https://www.youtube.com/watch?v=rK02eXm3mfI)  
[8] <a id="ref-8"></a> [youtube.com: Hugging Face Transformers - Intro to the Library](https://www.youtube.com/watch?v=jan07gloaRg)  
[9] <a id="ref-9"></a> [linkedin.com: How to Get Started with the Diffusers Library by Hugging Face: A Guide](https://www.linkedin.com/pulse/how-get-started-diffusers-library-hugging-face-guide-dushyant-kashyap-kkvuc)  
[10] <a id="ref-10"></a> [huggingface.co: Transformers Notebooks](https://huggingface.co/docs/transformers/notebooks)  
[11] <a id="ref-11"></a> [huggingface.co: Diffusers Documentation](https://huggingface.co/docs/diffusers/v0.21.0/index)  
[12] <a id="ref-12"></a> [huggingface.co: Transformers Documentation](https://huggingface.co/docs/transformers/index)  
[13] <a id="ref-13"></a> [huggingface.co: Transformers Documentation v4.15.0](https://huggingface.co/docs/transformers/v4.15.0/en/index)  
[14] <a id="ref-14"></a> [huggingface.co: Diffusers Training Overview](https://huggingface.co/docs/diffusers/v0.3.0/en/training/overview)  
[15] <a id="ref-15"></a> [huggingface.co: Transformers on the Hugging Face Hub](https://huggingface.co/docs/hub/en/transformers)  
[16] <a id="ref-16"></a> [github.com: Hugging Face Diffusers Repository](https://github.com/huggingface/diffusers)  
[17] <a id="ref-17"></a> [huggingface.co: Diffusers Basic Training](https://huggingface.co/docs/diffusers/en/tutorials/basic_training)  
[18] <a id="ref-18"></a> [huggingface.co: Diffusers Documentation](https://huggingface.co/docs/diffusers/en/index)  
[19] <a id="ref-19"></a> [huggingface.co: Diffusers on the Hugging Face Hub](https://huggingface.co/docs/hub/en/diffusers)  
[20] <a id="ref-20"></a> [huggingface.co: Diffusers Tutorials Overview](https://huggingface.co/docs/diffusers/en/tutorials/tutorial_overview)

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