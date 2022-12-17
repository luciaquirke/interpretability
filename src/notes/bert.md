https://arxiv.org/abs/1810.04805

### Background

During pre-training language models learn general language representations which are later adapted to specific tasks. The two main strategies for adapting  pre-trained LMs to downstream tasks are feature-based and fine-tuning. Feature-based approaches feed task inputs to the pre-trained model, then make its output available as one of the inputs to a downstream, task-specific model. Fine-tuning approaches add minimal task-specific parameters to  the pre-trained model then fine-tune all its parameters.

Usually these language models are unidirectional, where each token can only "pay attention" to the tokens which come before it in a sentence. This is a pretty arbitrary restriction and prevents the model from incorporating context in both directions. 

### BERT

- Multi-layer transformer encoder architecture
- Bidirectional self-attention

#### Input representation

- Sum token embeddings, segmentation embeddings and position embeddings, example:  
```
Token embeddings: [classification_token, my, dog, is, cute, separator_token, he, likes, play, ##ing, separator_token]
Segment embeddings: [A, A, A, A, A, A, B, B, B, B, B]
Position embeddings: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```
#### Pre-training

BERT uses a masked LM pre-training task to learn bidirectional encodings. When a sentence is input to BERT, each token in the sentence is processed using context from all the other tokens, not just the ones that came before it in the sentence. BERT also uses a sentence prediction task to pre-train text pair representations.

The next sentence prediction task uses sentence pairs. 50% of sentence pairs are adjacent, and 50% are randomly selected (non-adjacent). 
Seems like you could make this task more difficult by selecting sentence pairs that are close together (but still not adjacent) rather than random.

#### Fine-tuning

- Plug in downstream task's desired inputs and outputs
- Fine-tune all parameters

The whole fine-tuning process takes a few hours, in contrast with the long pre-training compute time.

### Glossary and Acronyms

Textual entailment - a directional relation between two text fragments, say t and h, where the truth (or not) of h can be inferred from t. 
- An example of positive textual entailment: t = all cats are female; h = the cat Mandy is a female. 
- An example of negative textual entailment: t = all cats are female; h = the cat Mandy is male.

LM - Language Model

### Concepts

- Transformers and self-attention
- GELU vs. RELU
