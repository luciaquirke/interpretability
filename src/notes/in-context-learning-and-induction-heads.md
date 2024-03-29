# In-context Learning and Induction Heads

* Investigates 1 and 2 layer attention-only transformers
* Found a common circuit called an "induction head"
* The induction head looks back over the current input sequence for previous instances of the current token (call it `A`), finds the token that came after it last time (call it `B`),
then predicts that the same completion will occur again (forming the sequence `A B . . . A -> B`).
* In the toy model each induction head was composed of two attention heads:
    * Previous token head in layer 1: copy information from the previous token into the next token
    * Main induction head in layer 2: use the information passed by the previous token head to find tokens preceded by (other instances of) the present token
* Induction heads appear to be the primary source of in-context learning in the toy model 
* We don't know how to prove that this is also the case for MLPs, but we can investigate by perturbing and studying the learning process and formation of network structures.
Then we can assemble an indirect case for what's going on in the MLPs.
dimensions of a multi-dimensional problems
* More complex circuits could be doing a more complex "fuzzy" or "nearest neighbour" version of this pattern completion, e.g. finding a pattern completion which is
similar in a few dimensions and using it to influence the predicted token (`A* B* . . . A -> B`, where `A* ~= A` and `B* ~= B`).

LQ: it feels intuitive to me that induction heads could be more than two layers, with each additional layer using the pattern matches of the previous one? Need to think about it more though, slightly shady

## In-context Learning

In-context learning is an emergent phenomenon in LLMs where tokens later in the context are easier to predict than tokens earlier in the context. This phenomenon can be used to get LLMs to perform specific tasks like translation or summary, e.g. give a model a few examples of french-to-english translations at query time then ask it to translate a new phrase, and its accuracy will be higher than if no example are provided. Because the LLM is not being trained at query time, this improvement shows that **LLM loss decreases at increasing token indices**.
* This doesn't seem exactly right to me - the previous tokens probably need to be relevant to decrease loss.

## Induction Heads

Induction heads are two layer circuits which find patterns in an input sequence and apply them to predict an output token. 

A simple induction head example: in the masked word prediction task `all cats are beautiful, all dogs are [mask]` an induction head could use the input sequence substring `are beautiful` to predict that masked token is `beautiful`, using direct copying.
A more abstract induction head example: in the masked word prediction task using the sequence `if all cats are green then why aren't [mask] kittens green` an induction head could use the `all cats are green` pattern to predict that the masked token is `all`, even though `cats` and `kittens` may only match in a few dimensions (e.g. species but not age) rather than being direct matches.

* Induction heads don't use encoded n-gram statistics, they apply general strategies like `if A B early in sequence, A -> B later in sequence` without regard for how statistically likely this is. This means they can work out of distribution in some sense, as long as the local distribution is consistent throughout the input sequence. LQ: I'm not sure how they know that n-gram statistics aren't encoded elsewhere in the model. 

## Experiment 

* A phase change happens early in the training for language models of all sizes, visible as a bump in the training loss. 
* During the phase change the difference in loss between tokens early and late in the sequence widens, showing that the majority of in-context learning is acquired here.
* Simultaneously, induction heads form within the model, capable of high-level/complex/fuzzy pattern completion.
* The paper attempts to establish a causal connection between these phenomena

## Results

If they perturb the transformer architecture in a way that causes the induction bump to occur in a different place in training, the formation of induction heads as well as in-context learning simultaneously move along with it. This doesn't fully establish cause, as they could both be formed by some other underlying factor. But with no other underlying factors emerging so far, they seem likely to be causal. There are several arguments in favor of the causal relationships:
* The correlation between phase change, in-context learning, and induction head formation
* That the perturbation which shifts the phase change also shifts the in-context learning and induction head formation
* When the induction heads are directly "knocked out" at test time in small models, the amount of in-context learning greatly decreases
* Empirical observations that seem to show heads detecting more complex/abstract matching patterns
* Direct mechanistic observations of induction heads working this way in small models
* It's intuitive that this could be extended to larger models and more abstract pattern matches
* Many phenomenon in this space scale smoothly from small to large models so it seems likely that induction heads would work the same way
The overall case is characterised as **circumstantial** but feels strong to me.

LQ: could we use synthetic data to prompt a transformer to learn a certain abstract connection in order to predict correctly? And compell this to be done with and without induction heads? Knocking out induction heads during training?

## Phase Changes

The language model phase change may be generally important. NN capabilities sometimes abruptly change as they train or scale, meaning dangerous NN behaviour can emergy abruptly too, e.g. violent reward hacking. If we could predict these phase changes or immediately shut down models when they happen we could improve safety.

> the phase change we observe forms an interesting potential bridge between the microscopic domain of interpretability and the macroscopic domain of scaling laws and learning dynamics.

### Glossary

Mechanistic interpretability: attempting to reverse engineer the detailed computations performed by a model

In-context learning: giving a LLM context on a task at query-time before asking the LLM to perform the task to reduce the loss, e.g. giving a model a few examples of french-to-english translations at query-time then asking it to translate a new phrase results in higher quality translations. 

Multilayer perceptron: fully connected feedforward neural network
Few shot learning: training a model to complete a task from only a few examples

Inductive reasoning: drawing conclusions by going from the specific to the general, e.g. every raven in a random sample of 3200 ravens is black -> this strongly supports the conclusion that all ravens are black.

Deductive reasoning: drawing conclusions by going from the general to the specific.

### Acronyms

MLP: multilayer perceptron
