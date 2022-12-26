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
  
Will add tests after the implementation proves useful.
  
I chose Bazel because I use it at work and it's super useful for Java + I wanted to learn more about it. It's good overall but doesn't have good support for specifying a hermetic Python interpreter - the Bazel IntelliJ plugin doesn't recognise the specified interpreter and raises an error for every package import. So I stuck with a non-hermetic venv interpreter - Python 3.9. 

The complex translation example has a bug where the pytorch dataset it uses has an [introduced error with a fix which hasn't been released yet](https://github.com/pytorch/text/issues/2001). I haven't fixed it yet because I don't know if I need it. If I need it I'll install a nightly build of Pytorch/Torchtext or versions from before the bug was introduced.

The matplotlib charts are hackily converted from the Altair plots because they're not that valuable for now. I might convert them to seaborn if I start making visualisations.

## Part 2: Design for induction heads replication
- [x] Read + write notes for Induction Heads paper
- [ ] Read + write notes for A Mathematical Framework for Transformer Circuits paper
- [ ] Draft design doc for induction heads implementation
- [ ] Investigate uncertainties in design doc
- [ ] Final design doc
