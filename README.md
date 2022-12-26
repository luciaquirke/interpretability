# interpretability

I'm attempting to replicate and extend the paper [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#:~:text=Induction%20heads%20are%20named%20by,Induction%20heads%20crystallize%20that%20inference.
).

Part 1: Get familiar with the transformer implementation
- [x] Read + write notes for Attention is All You Need and The Annotated Transformer
- [x] Run the existing Annotated Transformer implementation in google collab
- [x] Adapt it to run locally
  - [x] Extract classes into their own files
  - [x] Convert plots to matplotlib
  
I didn't get the complex translation example working because I don't know if I need it. The issue is that the pytorch dataset it uses has an [introduced error with a fix which hasn't been released yet](https://github.com/pytorch/text/issues/2001). If I need it I'll fix the example by installing a nightly build or a version from before the bug was introduced.

The matplotlib charts are hackily converted from the Altair plots because they're not that valuable for now. I might convert them to seaborn if I start making visualisations.

Part 2: Design for induction heads replication
- [x] Read + write notes for Induction Heads paper
- [ ] Read + write notes for A Mathematical Framework for Transformer Circuits paper
- [ ] Draft design doc for induction heads implementation
- [ ] Investigate uncertainties in design doc
- [ ] Final design doc
