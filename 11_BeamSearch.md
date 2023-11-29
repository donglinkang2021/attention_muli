# Beam Search

- Greedy Search
  - at any time step t, choose the token with the highest conditional probability from $\mathcal{Y}$

$$
y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}).
$$

- Exhaustive Search
  -  enumerate all the possible output sequences with their conditional probabilities, and then output the one that scores the highest predicted probability.
- Beam Search
  - At time step 1, we select the $k$ tokens with the highest predicted probabilities. Each of them will be the first token of $k$ candidate output sequences, respectively. 
  - At each subsequent time step, based on the $k$ candidate output sequences at the previous time step, we continue to select $k$ candidate output sequences with the highest predicted probabilities from $k\mathcal{Y}$ possible choices.