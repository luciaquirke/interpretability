# interpretability

I'm attempting to replicate and extend the paper [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#:~:text=Induction%20heads%20are%20named%20by,Induction%20heads%20crystallize%20that%20inference.
).

## Part 1: Get familiar with a transformer implementation
- [x] Read + write notes for Attention is All You Need and The Annotated Transformer
- [x] Run the existing Annotated Transformer implementation in google collab
- [x] Adapt it to run locally
  - [x] Set up bazel
  - [x] Extract classes out
  - [x] Convert plots to matplotlib
 - [ ] Work through Neel Nanda's transformer workbooks
 
### Annotated Transformer
  
Will add tests if the implementation proves useful. 

I chose Bazel because I use it at work and it's super useful for Java + I wanted to learn more about it. It's good overall but doesn't have good support for specifying a hermetic Python interpreter - the Bazel IntelliJ plugin doesn't recognise the specified interpreter and raises an error for every package import. So I stuck with a non-hermetic venv interpreter - Python 3.9. 

The complex translation example has a bug where the pytorch dataset it uses has an [introduced error with a fix which hasn't been released yet](https://github.com/pytorch/text/issues/2001). I haven't fixed it yet because I don't know if I need it. If I need it I'll install a nightly build of Pytorch/Torchtext or versions from before the bug was introduced.

The matplotlib charts are hackily converted from the Altair plots because they're not that valuable for now. I might convert them to seaborn if I start making visualisations.

### Neel Nanda workbooks and tutorials

https://colab.research.google.com/drive/1oGdXjrRcfSwXI0xJEu6oo-b1ygtXMsfj#scrollTo=cAudhHBe-TX8

The transformer workbook includes a cleaner implementation without the encoder-decoder structure and some useful libraries that I'll use going forward instead of modifying the Annotated Transformer implementation.
- einops
- easy transformer
- tqdm progress bar
And some new methods:
- [np.einsum](https://en.wikipedia.org/wiki/Einstein_notation)
- [torch.gather](https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms)
- torch.squeeze

## Part 2: Design for induction heads replication
- [x] Read + write notes for Induction Heads paper
- [ ] Read + write notes for A Mathematical Framework for Transformer Circuits paper
- [ ] Draft design doc for induction heads implementation
- [ ] Investigate uncertainties in design doc
- [ ] Final design doc

### Components for replication

* Per-token loss analysis:

> We start with a collection of models. (In our use, we'll train several different model architectures, saving dozens of “snapshots” of each over the course of training. We’ll use this set of snapshots as our collection of models.) Next, we collect the log-likelihoods each model assigns to a consistent set of 10,000 random tokens, each taken from a different example sequence. We combine these log-likelihoods into a "per-token loss vector" and apply Principal Component Analysis (PCA)

* Prefix matching score to detect induction heads by measuring attention heads' ability to perform the task we used to define induction heads: prefix matching then copying

* Smeared key parameter to allow one layer models to form induction heads
* "direct-path" logit attribution
