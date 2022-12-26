http://nlp.seas.harvard.edu/2018/04/03/attention.html

Supported by
https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
http://jalammar.github.io/illustrated-transformer/

# Attention Is All You Need breakdown

## Background

* Transformer architecture created to reduce sequential computation / increase parallelisation
* In older models the time complexity of relating individual tokens in the input grows linearly or even logarithmically as a function of the distance the tokens are from each other in the input format. Transformers make the operation complexity constant.
* The downside of the transformer is reduced effective resolution due to averaging attention weighted positions.
    * This downside is mitigated by an effect called Multi-Head Attention

Training data:
* 40 million English-German and English-French translation sentence pairs.
* Byte-pair encoding to reduce the wordsets to 37,000 and 32,000 tokens respectively.

## Architecture

Both the encoder and decoder are n=6 layers.

Each layer in the encoder has two sublayers. The first is a multi-head self attention mechanism, and the second is a fully connected feed-forward network. There's a residual connection around each sublayer.

The decoder is similar but it has a third sublayer, which performs multi-head attention over the output of the encoder stack. Some self-attention parameters are masked to prevent positions from attending to subsequent positions (uni-directional attention).

## Attention

A dot product compatibility function allows each query value to attend to each key/value pair in the dict. It's used in several ways. It uses queries and dict keys of dimension k, and dict values of dimension v.

A concrete example of its use is in "encoder-decoder attention" layers in the decoder. The queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. But what are queries, keys, and values, and why are their dimensions different? Well, a dict matrix with dimension [k * v] can be matrix multiplied with the queries, which have dimension [k * 1] (transpose(queries) * dict)

Attention(Q, K, V) = softmax(Q * transpose(K) / √k) * V

Or without the vectorised math, q * transpose(k) / √dim_k is a [1 * 1] constant because q and k are both vectors of length k, so every element in v gets multiplied by this constant.

> As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

### Encoding

Main article: ./positional-encoding.md

We use sine and cosine functions to produce a positional encoding matrix. If the input sequence is of length L our encoding matrix is [L * L], with each row representing the unique wave function for each position in a sequence of length L. We add this positional encoding matrix to our symbol embedding to get our overall embedding.

## Why self-attention


### Glossary

Self attention: also called intra-attention. An attention mechanism relating different positions of a sequence (input sequence or mid-processing sequence) in order to compute a representation of the sequence. I imagine an example could be a circuit which processes a sentence of words (tokens) to relate adjective tokens to their subject tokens.

Transduction: or transductive inference. Reasoning from observed, specific (training) cases to specific (test) cases. Contrasts with induction, where specific (training) cases are used to learn generalisations which are then applied to specific (test) cases. The difference is that transduction can use the structure of the unlabelled (test) data in its reasoning, whereas with induction the generalisations are pre-determined and appled without using this additional information. A canon transductive model is k-nearest neighbours, which uses the training set directly to label each new piece of test data.

Transducer: NLP sequence prediction task model (e.g. for translation or sequence tagging). The root word "transduction" is here applied in a different and much looser sense than in transductive inference.

Sequence model: any model that inputs and output sequences of data. A transducer is always a sequence model. A transduction model may also be a sequence model, although KNN is a counter-example.

Auto-regressive model: model where output tokens are produced one at a time, using the corresponding input token and all previous input tokens. So the first output token is predicted using the first input token, the second output token is predicted using the first two input tokens, etc.

Byte pair encoding:
* Original meaning: greedy data compression algorithm where the most common pair of consecutive bytes is replaced with a byte which doesn't occur in the data. This process is then repeated on the new data to increase compression. A hashmap of { replacement bytes -> input data indices} can be used to decompress the data.
* ML meaning: modified version of the original algorithm for text encoding, where frequently occuring subwords are merged together. Also called subword-based tokenization. e.g., the rare word athazagoraphobia could be split up into the more frequent subwords ['▁ath', 'az', 'agor', 'aphobia'].

Compatibility function: maps from `X` and `W` (i.e. `theta`) to `Y`. An example of a compatibility function is `hypothesis(theta) = X*theta`

Desiderata: something that is desired

Long range dependency: when one word in a sequence is relevant to understanding another word, but they're separated by many words.

**Related things:**
Recurrent attention
Sequence-aligned recurrence/RNNs
End-to-end memory network

