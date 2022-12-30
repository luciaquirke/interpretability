# In-context Learning and Induction Heads

## Summary

* 1 and 2 layer attention-only transformers (toy model).
* Found a common circuit in the model, named "induction head"
* "Context" is the input sequence to the model plus the previously produced tokens in the output sequence.
* The model's induction heads look back over the context for previous instances of the current token (call it
`A`), find the token that came after it last time (call it `B`), then predict that the same completion will occur again 
(forming the sequence `A B . . . A -> B`).
* Hypothesis: more complex induction heads in big LLMs do something more general: identify certain kinds of relevant 
content from earlier in the context
    * Possible definition: Induction heads are circuits which find patterns in the context and apply them to predict 
  an output token.
* Induction heads form at the same steps of LLM training that the phase change (the little bump) in the loss curve occurs
* Removing induction heads from the toy model removes most in-context learning as measured by decreasing loss at each 
subsequent output token.
    * Calculated as `loss of 500th token in the context - average loss of 50th token in the context`, averaged over 
  database examples 
    * Paper makes indirect case that this applies to LLMs as well as the toy model

## What are induction heads?

An attention head is a parallel attention layer which takes a fraction of the previous layer as input 
(see The Annotated Transformer). Each induction head is composed of two attention heads:
* Previous token head in layer 1: copy information from the previous token into the next token
* Main induction head in layer 2: use the information passed by the previous token head to find tokens preceded by 
      (other instances of) the present token

Induction heads don't use encoded n-gram statistics, they apply general strategies like `if A B early in context,
A -> B later in context` without regard for how statistically likely this is. This means they can work out of
distribution in some sense, as long as the local distribution is consistent throughout the context. Note n-gram
statistics may be encoded elsewhere in the model, they're just not used by the induction heads which only look at the
current context.

A simple induction head application: in the sequence completion task `the cat sat on the mat, the cat ...` an 
induction head would promote the completion `sat on the mat` using direct copying.

A more abstract induction head example: in the masked word prediction task `all cats are green, 
[mask] kittens are green`, an induction head could use the `all cats are green` pattern to predict that the 
masked token is `all`, even though `cats` and `kittens` may only match in a few dimensions (e.g. species but not age) 
rather than being direct matches.

Induction heads are defined narrowly for this paper, and only match example 1. But there are examples of induction
heads acting like in example 2, and the circuit found in the toy model could be simply extended to facilitate example
2.

Formally, an induction head is an attention head which exhibits the following two properties on a repeated random 
sequence 6 of tokens:
* Prefix matching: The head attends back to previous tokens that were followed by the current and/or recent tokens. 
That is, it attends to the token which induction would suggest comes next.
* Copying: The head’s output increases the logit corresponding to the attended-to token.

## How might induction heads be responsible for most in-context learning in MLPs too?

More complex circuits could be doing a more complex "fuzzy" or "nearest neighbour" version of the pattern completion 
induction heads demonstrate in the toy model, e.g. finding a pattern completion which is similar in a few dimensions 
and using it to influence the predicted token (`A* B* . . . A -> B`, where `A* ~= A` and `B* ~= B`).

LQ: it feels intuitive that induction heads could be more than two layers, with each additional layer using the 
pattern matches of the previous one? Come back to this.

## Definition of in-context Learning

In-context learning is an emergent phenomenon in LLMs where tokens later in the context are easier to predict than 
tokens earlier in the context. This phenomenon can be used to get LLMs to perform specific tasks like translation or 
summary, e.g. give a model a few examples of french-to-english translations at query time then ask it to translate a new 
phrase, and its accuracy will be higher than if no example are provided. Because the LLM is not being trained at query 
time, this improvement shows that **LLM loss decreases at increasing token indices**.

## Experiment 

* A phase change happens early in the training for language models of all sizes, visible as a bump in the training loss.
   * > The phase change is the only place in training where the loss is not convex (monotonically decreasing in slope).
   * Each point on the loss curve is averaged over thousands of predicted tokens so improvements that only apply to a 
  small subset of examples (e.g. learning arithmetic) won't make much difference to the curve. The bump has to be caused
  by a widespread, major change in behaviour.
   * For small models the phase change is the first loss curve divergence between 1 layer and multi layer models 
* During the phase change the difference in loss between tokens early and late in the sequence widens, showing that the 
majority of in-context learning is acquired here.
* Simultaneously, induction heads form within the model, capable of high-level/complex/fuzzy pattern completion.
* The paper attempts to establish a causal connection between these phenomena

## Results

If they perturb the transformer architecture in a way that causes the induction bump to occur in a different place in 
training, the formation of induction heads as well as in-context learning simultaneously move along with it. This 
doesn't fully establish cause, as they could both be formed by some other underlying factor. But with no other 
underlying factors emerging so far, they seem likely to be causal. There are several arguments in favor of the causal 
relationships:
* The correlation between phase change, in-context learning, and induction head formation
* That the perturbation which shifts the phase change also shifts the in-context learning and induction head formation
    * LQ: what kind of perturbation accomplishes this?
    * LQ: this applies to large models but the time resolution has to be lower for performance reasons. Maybe could 
  replicate with slightly larger models - 4-5 layers.
* When the induction heads are directly "knocked out" at test time in small models, the amount of in-context learning greatly decreases
    * LQ: what knock out algorithm? See reference 
    * We can't simply ablate attention heads to remove in-context learning in MLPs because the induction head could be
          formed of an attention layer + an MLP layer instead
    * We knock out one induction head at a time so if two are doing the same thing we won't see performance drop
* Empirical observations that seem to show heads detecting more complex/abstract matching patterns
    * they find examples that perform analogous matching like example 2 above, while ALSO performing simple prefix matching
  and copying
* Direct mechanistic observations of induction heads working this way in small models. 
* It's intuitive that this could be extended to larger models and more abstract pattern matches
* Many phenomenon in this space scale smoothly from small to large models so it seems likely that induction heads would work the same way
The overall case is characterised as **circumstantial** but feels very promising.

LQ: could we use synthetic data to prompt a transformer to learn a certain abstract connection in order to predict 
correctly? And compell this to be done with and without induction heads? Knocking out induction heads during training?

## Alternate hypotheses for improvements during phase change

* Maybe the phase change is the point at which the model generally learns how to compose layers, enabling both
induction heads and other useful things that require composing multiple layers
* Or some other more general thing which enables induction heads 
* In-context learning is roughly constant (0.4 nats) post phase change but other mechanisms could be kicking in to 
keep it constant even as accuracy earlier in the context improves. E.g. if the model learns something which only helps
it early in the context, and it simultaneously learns something else which only helps it late in the context, we'd see 
no improvement in in-context learning as measured by decreasing loss late in the context relative to early. But why
would the two strategies be learned simultaneously? This seems like an unpromising hypothesis.
* Induction heads could evolve into something with different properties during training, becoming something quite different
by the end. Then what we think of as "induction heads" would not be responsible for in-context learning at test time.


## Phase Changes and AI Safety

The language model phase change may be generally important. NN capabilities sometimes abruptly change as they train or 
scale, meaning dangerous NN behaviour can emergy abruptly too, e.g. violent reward hacking. If we could predict these 
phase changes or immediately shut down models when they occur we could improve safety.

> the phase change we observe forms an interesting potential bridge between the microscopic domain of interpretability 
> and the macroscopic domain of scaling laws and learning dynamics.
 
LQ: makes me wonder why all the induction heads are learned in the same few training steps. is there a mechanism for 
propagating useful adaptations across the model, or are they forming independently?

In-context learning makes it harder to predict how a model will behave after a long context. We see this in the short
term with people using lengthy prompts to hack Chat GPT's behavioural conditioning. In the long term in-context learning
means behaviours learned via mesa-optimisation could be triggered at test time.

### Glossary

* Mechanistic interpretability: attempting to reverse engineer the detailed computations performed by a model
* Context: an input sequence to a model + the already generated parts of the output sequence
* In-context learning: giving a LLM context on a task at query-time before asking the LLM to perform the task to reduce 
the loss, e.g. giving a model a few examples of french-to-english translations at query-time then asking it to translate 
a new phrase results in higher quality translations. 
* Multilayer perceptron: fully connected feedforward neural network
* Few shot learning: training a model to complete a task from only a few examples
* Inductive reasoning: drawing conclusions by going from the specific to the general, e.g. every raven in a random sample 
of 3200 ravens is black -> this strongly supports the conclusion that all ravens are black.
* Deductive reasoning: drawing conclusions by going from the general to the specific.
* Nat: the natural unit of information, obtained by using the natural logarithm `ln` instead of the base 2 logarithm
`lg` when defining entropy and related information theoretic functions. When `lg` is used instead the unit is bits.
    * logs are in a similar category to roots; they do the inverse computation of a regular operator. For roots it's
  powers, for logs it's exponentials. E.g. if you want to rearrange x = 2^y to isolate y, you can go y = log_base_2(x)
    * 
### Acronyms

* MLP: multilayer perceptron
* LLM: large language model
