# Transformer Notes

## Embedding and Transformer Inputs

```
words = [a, b, c]
embedding = {
    "a": [1,0,0],
    "b": [0,1,0],
    "c": [0,0,1]
}
```

An embedding is basically a lookup table to convert words to vectors that ML models can use. The simple embedding above 
is sometimes called one-hot encoding. No structure is feature engineered into the embedding, so the model considers each 
label independently. 

You could argue that for encoding things with obvious structure, like letters of the alphabet, we
**want** to add structure to the data. But this won't work for more items with more complex structure, and it turns out 
our model can determine structure for itself anyway.

A dimension is something which varies independently.

The "lookup table" process is multiplying a fixed matrix with a one-hot encoded vector to select the matrix column 
corresponding to the word encoded in the vector.

We want to convert arbitrary text to integers in a bounded range. 
* Idea 1: use a dictionary and convert each word to integers. **advantages** simple. **disadvantages** can't cope with 
arbitrary text, misspellings, URLs, punctuation, new words etc.
* Idea 2: use characters! **advantages** fixed and limited number of characters, can do arbitrary text. **disadvantages**
inefficient because some character sequences are much more meaningful than others, see "fn,ilhbds" vs. "language"
  * This seems pretty sus because you could say the same thing about words or subwords. Seems like foreshadowing for 
  feature engineering
* **Idea 3**: byte pair encoding, which creates a subword to integer mapping starting from the ASCII character set and 
working its way up through more and more common subwords until it hits a fixed mapping size. **advantages** conveys 
strictly more information than just using characters. **disadvantages** sometimes tokenises words in varied and cursed
ways depending on whether they're capitalised, are prefixed with whitespace etc. Arithmetic is a total mess: length is
inconsistent, common numbers are bundled together

Now with byte pair encoding:

```
words = [a, b, c]
byte_pair_encodings = {
    "a": 21052,
    "b": 31248,
    "c": 04938
}
embedding = {
    21052: [1,0,0],
    31248: [0,1,0],
    04938: [0,0,1]
}
```

## Transformer outputs

A tensor is generalised from the sequence scalar, vector, matrix. So a scalar is a tensor of rank 0, and a matrix
is a tensor of rank 2.

The model outputs a tensor of logits - one vector of size len(subword vocab) for each input token. We use softmax to 
convert each vector of logits to a distribution, and take the token with the highest probability as our most likely
next token for each input token. We can add the predicted next token at the final token to the end of our input and 
re-run to continue predicting.

