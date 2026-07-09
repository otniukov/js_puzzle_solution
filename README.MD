## Recovering Residual Blocks Order via Geometric Reconstruction and Local Optimization

This repository contains a solution to the  [*Jane Street ML puzzle*](https://huggingface.co/spaces/jane-street/droppedaneuralnet).

The objective is to recover the original ordering of the residual blocks in a trained neural network. The provided model consists of 48 residual blocks with identical hidden dimensions followed by a final prediction layer. For every layer, the weight matrices and bias vectors are available, together with a dataset containing input features, target values, and the model's predictions.

The proposed solution exploits two complementary properties of well-trained residual networks. First, neighboring residual blocks approximate a smooth trajectory in representation space, allowing the global block order to be reconstructed using geometric similarity. Second, any remaining ordering errors can be corrected by directly optimizing the network's prediction error.

The reconstruction proceeds in three steps:

1. **Pair residual layers** to recover the original residual blocks.
2. **Recover an initial ordering** using the smoothness of the induced vector fields.
3. **Refine the ordering** through local optimization with adjacent block swaps.

Conceptually, the algorithm consists of two optimization stages:

1. **Geometric reconstruction**, which maximizes the smoothness of the recovered representation trajectory.
2. **Functional refinement**, which minimizes the prediction error of the reconstructed network.

Despite relying only on simple heuristics, this procedure is sufficient to recover the original block ordering exactly.


## Pairing Residual Blocks

The model contains 97 linear layers. One of them is the final prediction layer with shape $[48, 1]$, while the remaining 96 layers form the 48 residual blocks.

The model is well trained based on given dataset and MSE loss on predictions and truths. So, residual connections create gradient coherence. 

Previous work has shown that residual MLPs often exhibit low-rank autoencoder-like behavior. Motivated by this observation, I assume that the encoder and decoder within a residual block become approximately symmetric. Under this assumption, the encoder weight matrix and the transpose of the decoder weight matrix should be similar.

To quantify this property, a **hidden alignment** score is defined for each candidate input–output layer pair.
For a candidate pair of weight matrices, the statistic

$$cos(Win[k,:],Wout[:,k])$$ 

measures how well the (k)-th hidden neuron satisfies this encoder–decoder symmetry. Aggregating these similarities over all hidden neurons yields a single alignment score for the layer pair. 

Given the sets of candidate input layers, $W_{\text{in}}$, and output layers, $W_{\text{out}}$, the alignment matrix $A \in \mathbb{R}^{N_{\text{in}} \times N_{\text{out}}}$ is computed, where 

$$A_{ij} = \text{HiddenAlignment}\left(W_{\text{in}}^{(i)},W_{\text{out}}^{(j)}\right).$$


For each input layer, the output layer with the highest alignment score is selected, $j^* = \arg\max_j A_{ij}$, forming a matched input–output layer pair. These pairs are then converted into residual blocks by combining the corresponding metadata from the input and output layer lists.



## Initial Block Ordering
Recovering the block ordering directly is a combinatorial optimization problem with $48!$ possible permutations. An exhaustive search is therefore infeasible.

Instead, the reconstruction is divided into two stages. First, a geometry-based heuristic recovers an approximate ordering by exploiting the continuous dynamics of residual networks. This ordering is then refined through local optimization using the network prediction loss.

A residual network can be interpreted as a discretization of a continuous dynamical system. Since each residual block performs the update

$$x_{k+1}=x_k+f_k(x_k),$$

the network approximates the ordinary differential equation


$$\frac{dx}{dt}=v(x,t),$$

where each residual block approximates the vector field at a particular time.

Under this interpretation, adjacent residual blocks correspond to nearby time steps and therefore should induce similar vector fields,

$$f_i \approx f_{i+1}.$$

To quantify this property, a **pairwise similarity matrix** is computed using the residual updates produced by each block:

$$S_{ij} = \mathbb{E}_x\left[\cos\!\left(h_i(x),\,h_j(x)\right)\right],$$

where

$$h_i(x_0) = B_i(x_0) - x_0$$

is the residual update of block $i$ evaluated on the raw input features $x_0$, rather than on the intermediate activations encountered during a forward pass.

The similarity matrix reveals two distinctive outlier patterns.

The first pattern exhibits uniformly low similarity to all other blocks. Its highest similarity score is only $0.107$, substantially lower than typical values observed elsewhere in the matrix. This block is hypothesized to be the final residual block, immediately preceding the prediction head. In well-trained ResNets, the representation just before the output layer is often highly specialized and therefore substantially different from the preceding hidden representations.

The second pattern identifies two blocks $21$ and $34$ with strong mutual similarity of approximately $0.81$, making both plausible candidates for the beginning of the network. Since the similarity metric evaluates every block on the same raw input $x_0$, rather than on the intermediate activations encountered during a forward pass, the first residual block is expected to exhibit a distinctive similarity pattern.

Because the first residual block has only one adjacent neighbor in the true ordering, it is expected to exhibit a weaker secondary similarity than an interior block. Block $21$ matches this pattern more closely than block $34$: its second-best similarity score is noticeably lower, indicating a weaker connection to the remainder of the similarity graph. Consequently, block $21$ is selected as the starting point for the greedy traversal.

These observations identify plausible candidates for the first and last residual blocks. Starting from the estimated first block, an initial ordering is constructed using a 
greedy nearest-neighbor traversal, where each subsequent block is chosen as the most similar unvisited block according to the similarity matrix.


## Ordering Refinement

The greedy reconstruction provides a good initial ordering, but it is not exact. 
For a candidate ordering, the objective is the mean squared error $MSE$ between the predictions of the reconstructed model and the predictions provided with the puzzle. Starting from the greedy pre-ordering, the optimization seeks the permutation that minimizes this loss.

The optimization is motivated by the *Neural ODE* interpretation introduced in the previous section. Since neighboring residual blocks approximate nearby vector fields, the greedy pre-ordering is expected to recover the correct global trajectory, leaving only a small number of local ordering errors. Consequently, only **adjacent block** swaps are considered during optimization. This dramatically reduces the search space while remaining sufficient to correct the remaining inconsistencies.

The optimization itself is inspired by local-improvement heuristics for the Traveling Salesman Problem. During each iteration, every adjacent pair of blocks is swapped in turn, and the resulting $MSE$ is evaluated. A swap is accepted if it strictly decreases the loss; otherwise, it is rejected. After completing a forward sweep through the ordering, the same procedure is repeated in the reverse direction. The process terminates when an entire iteration produces no improving swaps or when the maximum number of iterations is reached.

This optimization can be viewed as a hill-climbing procedure over the space of block permutations. Because the initialization already captures the global structure of the network, only a handful of local corrections are required to recover the exact ordering. In practice, this procedure consistently converges to the correct block sequence, yielding zero prediction MSE on the provided dataset.

## Discussion

Although the final optimization relies on the network prediction loss, the geometric pre-ordering already recovers most of the original architecture. The initial ordering achieves a Spearman rank correlation above $0.9$ with the ground-truth block sequence, suggesting that the pairwise similarity matrix captures much of the underlying continuous dynamics.

It is worth noting that the similarity metric is only an approximation of the true network dynamics. The residual update

$$h_i(x_0) = B_i(x_0) - x_0$$

is evaluated on the raw input features $x_0$, whereas the actual network applies the (i)-th residual block to the intermediate representation $x_{i-1}$. Consequently, the measured similarities do not exactly correspond to the vector fields traversed during inference. Despite this simplification, the approximation is sufficiently informative to recover an ordering that is already very close to the original.

Another interesting observation is the robustness of the pairing stage. The hidden-alignment score assumes an approximate encoder–decoder symmetry within each residual block, even though the network contains *ReLU* nonlinearities. In principle, *ReLU* could break this symmetry by activating different subsets of neurons for different inputs. Nevertheless, the learned weights appear to preserve enough of the underlying linear structure for the alignment metric to remain highly discriminative.

These observations suggest that the geometric organization of well-trained residual networks is considerably more structured than might be expected. Exploiting this structure makes it possible to recover the original block ordering with only simple greedy heuristics followed by local optimization.
