---
layout: distill
title: An Illustrated Guide to Automatic Sparse Differentiation
description: Your blog post's abstract.
  Please add your abstract or summary here and not in the main body of your text. 
  Do not include math/latex or hyperlinks.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Build page via 
# ```bash
# rbenv init
# rbenv install 3.3.0
# rbenv local 3.3.0
# bundle install
# bundle exec jekyll serve --future --open-url /blog/sparse-autodiff/ --livereload
# ```
#
# Then navigate to `/blog/sparse-autodiff/`

# Anonymize when submitting
# authors:
#   - name: Anonymous

# TODO before submission:
# - revert CI workflows
# - check correct figure caption numbering and references
# - check accessibility – color-blindness
# - check correct rendering of SVGs on multiple browsers

authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-sparse-autodiff.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Automatic differentiation
    subsections:
    - name: The chain rule
    - name: AD is matrix-free
    - name: Forward-mode AD
    - name: Reverse-mode AD
  - name: Automatic sparse differentiation
    subsections:
    - name: Sparse matrices
    - name: Leveraging structure
    - name: Sparsity pattern detection and coloring
  - name: Pattern detection
    subsections:
    - name: Compressing Jacobians
    - name: Propagating index sets
    - name: Alternative evaluation
  - name: Matrix coloring
  - name: Second-order sparse differentiation
  - name: Demonstration

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: ##bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

<!-- LaTeX commands -->
<div style="display: none">
    $$
    \newcommand{\colorf}[1]{\textcolor{RoyalBlue}{#1}}
    \newcommand{\colorh}[1]{\textcolor{RedOrange}{#1}}
    \newcommand{\colorg}[1]{\textcolor{PineGreen}{#1}}
    \newcommand{\colorv}[1]{\textcolor{VioletRed}{#1}}
    \def\sR{\mathbb{R}}
    \def\vx{\mathbf{x}}
    \def\vv{\mathbf{v}}
    \def\vb{\mathbf{e}}
    \newcommand{\vvc}[1]{\colorv{\vv_{#1}}}
    \newcommand{\vbc}[1]{\colorv{\vb_{#1}}}
    \newcommand{\dfdx}[2]{\frac{\partial f_{#1}}{\partial x_{#2}}(\vx)}
    \newcommand{\J}[2]{J_{#1}(#2)} 
    \def\Jf{\J{f}{\vx}}
    \def\Jg{\J{g}{\vx}}
    \def\Jh{\J{h}{g(\vx)}}
    \def\Jfc{\colorf{\Jf}}
    \def\Jgc{\colorg{\Jg}}
    \def\Jhc{\colorh{\Jh}}
    \newcommand{\D}[2]{D{#1}(#2)}
    \def\Df{\D{f}{\vx}}
    \def\Dg{\D{g}{\vx}}
    \def\Dh{\D{h}{g(\vx)}}
    \def\Dfc{\colorf{\Df}}
    \def\Dgc{\colorg{\Dg}}
    \def\Dhc{\colorh{\Dh}}
    $$
</div>

First-order optimization is ubiquitous in Machine Learning (ML) but second-order optimization is much less common.
The intuitive reason is that large gradients are cheap, whereas large Hessian matrices are expensive.
Luckily, in numerous applications of ML to science or engineering, **Hessians (and Jacobians) exhibit sparsity**:
most of their coefficients are known to be zero.
Leveraging this sparsity can vastly **accelerate Automatic Differentiation** (AD) for Hessians and Jacobians,
while decreasing its memory requirements.
Yet, while traditional AD is available in many high-level programming languages,
**automatic sparse differentiation (ASD) is not as widely used**.
One reason is that the underlying theory was developed outside of the ML research ecosystem,
by people more familiar with low-level programming languages.

With this blog post, we aim to shed light on the inner workings of ASD,
thus bridging the gap between the ML and AD communities.
We start out with a short introduction to traditional AD,
covering the computation of Jacobians in both forward and reverse mode.
We then dive into the two primary components of ASD:
**sparsity pattern detection** and **matrix coloring**.
Having described the computation of sparse Jacobians,
we move on to sparse Hessians.  
We conclude with a practical demonstration of ASD,
providing performance benchmarks and guidance on when to use ASD over AD.

## Automatic Differentiation

Let us start by covering the fundamentals of traditional AD.

AD makes use of the **compositional structure** of mathematical functions like deep neural networks.
To make things simple, we will mainly look at a differentiable function $f$
composed of two differentiable functions
$g: \sR^{n} \rightarrow \sR^{p}$ and $h: \sR^{p} \rightarrow \sR^{m}$,
such that $f = h \circ g: \sR^{n} \rightarrow \sR^{m}$.
The insights gained from this toy example should translate directly to more deeply composed functions $f = g^{(L)} \circ g^{(L-1)} \circ \cdots \circ g^{(1)}$.
For ease of visualization, we work in small dimension, but the real benefits of ASD only appear as the dimension grows.

### The chain rule

For a function $f: \sR^{n} \rightarrow \sR^{m}$ and a point of linearization $\vx \in \sR^{n}$,
the Jacobian $J_f(\vx)$ is the $m \times n$ matrix of first-order partial derivatives, such that the $(i,j)$-th entry is

$$ \big( \Jf \big)_{i,j} = \dfdx{i}{j} \in \sR \quad . $$

For a composed function 

$$ \colorf{f} = \colorh{h} \circ \colorg{g}, $$

the **multivariate chain rule** tells us that we obtain the Jacobian of $f$ by **multiplying** the Jacobians of $h$ and $g$:

$$ \Jfc = \Jhc \cdot \Jgc \quad .$$

Figure 1 illustrates this for $n=5$, $m=4$ and $p=3$.
We will keep using these dimensions in following illustrations.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/chainrule_num.svg" class="img-fluid" %}
<div class="caption">
    Figure 1: Visualization of the multivariate chain rule for $f = h \circ g$.
</div>

### AD is matrix-free

We have seen how the chain rule translates the compositional structure of a function into the product structure of its Jacobian.
Thanks to the small dimensions $n$, $m$ and $p$, this approach worked well on our toy example in Figure 1.
In practice however, there is a problem:
**materializing intermediate Jacobian matrices is inefficient and often impossible**, especially with a dense matrix format.
Examples of dense matrix formats include NumPy's `ndarray`, PyTorch's `Tensor`, JAX's `Array` and Julia's `Matrix`.

As a motivating example, let us take a look at a tiny convolutional layer.
We consider a convolutional filter of size $5 \times 5$, a single input channel and a single output channel.
An input of size $28 \times 28 \times 1$ results in a $576 \times 784$ Jacobian, the structure of which is shown in Figure 2.
All the white coefficients are **structural zeros**.

If we materialize the entire Jacobian as a dense matrix:

- we waste time computing coefficients which are mostly zero;
- we waste memory storing those zero coefficients.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/big_conv_jacobian.png" class="img-fluid" %}
<div class="caption">
    Figure 2: Structure of the Jacobian of a tiny convolutional layer.
</div>

In modern neural network architectures, which can contain over one trillion parameters,
computing intermediate Jacobians is not only inefficient: it exceeds available memory.
AD circumvents this limitation using **linear maps**, lazy operators that act exactly like matrices but without materializing them.

The differential $Df: \vx \longmapsto Df(\vx)$ is a linear map which provides the best linear approximation of $f$ around a given point $\vx$.
We can rephrase  the chain rule as a **composition of linear maps** instead of a product of matrices:

$$ \Dfc = \colorf{\D{(h \circ g)}{\vx}} = \Dhc \circ \Dgc .$$

Note that all terms in this formulation of the chain rule are linear maps.
A new visualization for our toy example can be found in Figure 3b.
Our illustrations distinguish between materialized matrices and linear maps by using solid and dashed lines respectively.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/chainrule_num.svg" class="img-fluid" %}
<div class="caption">
    Figure 3a: Chain rule using materialized Jacobians (solid outline).
</div>

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/matrixfree.svg" class="img-fluid" %}
<div class="caption">
    Figure 3b: Chain rule using matrix-free linear maps (dashed outline).
</div>

*We visualize "matrix entries" in linear maps to build intuition.
Even though following illustrations will sometimes put numbers onto these "matrix entries",
linear maps are best thought of as black-box functions.*

### Forward-mode AD

Now that we have translated the compositional structure of our function $f$ into a compositional structure of linear maps, we can evaluate them by propagating **materialized vectors** through them.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode_eval.svg" class="img-fluid" %}
<div class="caption">
    Figure 4: Evaluating linear maps in forward-mode.
</div>

Figure 4 illustrates the propagation of a vector $\vv_1 \in \sR^n$ from the right-hand side.
Since we propagate in the order of the original function evaluation, this is called **forward-mode AD**.

In the first step, we evaluate $Dg(\vx)(\vv_1)$.
Since this operation by definition corresponds to 

$$ \vvc{2} = \Dgc(\vvc{1}) = \Jgc \cdot \vvc{1} \;\in \sR^p ,$$

it is also commonly called a **Jacobian-vector product** (JVP) or **pushforward**.
The resulting vector $\vv_2$ is then used to compute the subsequent JVP 

$$ \vvc{3} = \Dhc(\vvc{2}) = \Jhc \cdot \vvc{2} \;\in \sR^m ,$$

which in accordance with the chain rule is equivalent to 

$$ \vvc{3} = \Dfc(\vvc{1}) = \Jfc \cdot \vvc{1} ,$$

the JVP of our composed function $f$.

**Note that we did not materialize intermediate Jacobians at any point** – we only propagated vectors through linear maps.

### Reverse-mode AD

We can also propagate vectors through our linear maps from the left-hand side, 
resulting in **reverse-mode AD**, shown in Figure 5.
It is also commonly called a **vector-Jacobian product** (VJP) or **pullback**.
Just like forward-mode, reverse-mode is also matrix-free: **no intermediate Jacobians are materialized at any point**.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/reverse_mode_eval.svg" class="img-fluid" %}
<div class="caption">
    Figure 5: Evaluating linear maps in reverse-mode.
</div>

<!-- TODO: should we add more text here? Not sure anything would be gained and it's notationally bothersome. -->

### From linear maps back to Jacobians

The linear map formulation allows us to avoid intermediate Jacobian matrices in long chains of function compositions.
But can we use this machinery to materialize the **Jacobian** of the composition $f$ itself?

As shown in Figure 6, we can **materialize Jacobians column by column** in forward mode.
Evaluating the linear map $Df(\vx)$ on the $i$-th standard basis vector materializes the $i$-th column of the Jacobian $J_f(\vx)$:

$$ \Dfc(\vbc{i}) = \left( \Jfc \right)_\colorv{i,:} $$

Thus, materializing the full $m \times n$ Jacobian requires one JVP with each of the $n$ standard basis vectors of the **input space**.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode.svg" class="img-fluid" %}
<div class="caption">
    Figure 6: Forward-mode AD materializes Jacobians column-by-column.
</div>

As illustated in Figure 7, we can also **materialize Jacobians row by row** in reverse mode.
Unlike forward mode in Figure 6,
this requires one VJP with each of the $m$ standard basis vectors of the **output space**.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/reverse_mode.svg" class="img-fluid" %}
<div class="caption">
    Figure 7: Reverse-mode AD materializes Jacobians row-by-row.
</div>

Since neural networks are usually trained using scalar loss functions,
reverse-mode AD only requires the evaluation of a single VJP to compute a gradient.
This makes it the method of choice for machine learners,
who typically refer to reverse-mode AD as *backpropagation*.

## Sparse automatic differentiation

### Sparse matrices

Sparse matrices are matrices in which most elements are zero.
We refer to linear maps as "sparse linear maps" if they materialize to sparse matrices.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_matrix.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_map.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 8: A sparse Jacobian and its corresponding sparse linear map.
</div>

Whene functions have many inputs and many outputs,
a given output does not always depend on every single input.
This endows the corresponding Jacobian with a sparse structure,
where zero coefficients denote an absence of (first-order) dependency.
The previous case of a convolutional layer is a simple example.
An even simpler example is an activation function applied elementwise,
for which the Jacobian is the identity matrix.

### Leveraging structure

Assuming we know the structure of the Jacobian, we can find orthogonal, 
non-overlapping columns or rows via a method called **matrix coloring** that we will go into more detail on later.

**The core idea of ASD is that we can materialize multiple orthogonal columns or rows in a single evaluation.**
Since linear maps are additive, it always holds that

$$ \Dfc(\vbc{i}+\ldots+\vbc{j}) 
= \underbrace{\Dfc(\vbc{i})}_{\left( \Jfc \right)_\colorv{i,:}} 
+ \ldots
+ \underbrace{\Dfc(\vbc{j})}_{\left( \Jfc \right)_\colorv{j,:}} 
. $$

The right hand side summands each correspond to a column of the Jacobian.
If the columns are **orthogonal** and their **structure is known**, 
the sum can be decomposed into its summands, materializing multiple columns in a single JVP.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad.svg" class="img-fluid" %}
<div class="caption">
    Figure 9: Materializing multiple orthogonal columns of a Jacobian in forward-mode.
</div>

This specific example using JVPs corresponds to sparse forward-mode AD 
and is visualized in Figure 9, where all orthogonal columns have been colored in matching hues.
By computing a single JVP with the vector $\mathbf{e}_1 + \mathbf{e}_2 + \mathbf{e}_5$, 
we materialize the sum of the first, second and fifth column of our Jacobian.

Since we can assume we know the structure of the Jacobian,
we can assign the values in the resulting vector to the correct Jacobian entries.
The full forward-mode ASD materialization of our toy Jacobian is shown in Figure X. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad_forward_full.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad_forward_decompression.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure X: Materializing a Jacobian with forward-mode ASD: (a) compressed evaluation of orthogonal columns (b) decompression to Jacobian matrix
</div>

The same idea can also be applied to reverse mode AD, as shown in Figure Y.
Instead of finding orthogonal column, we need to find orthogonal rows.
We can then materialize multiple rows in a single VJP.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad_reverse_full.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparse_ad_reverse_decompression.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure Y: Materializing a Jacobian with reverse-mode ASD: (a) compressed evaluation of orthogonal rows (b) decompression to Jacobian matrix
</div>

### Sparsity pattern detection and coloring

Unfortunately, our initial assumption had a major flaw: 
Since AD only gives us a composition of linear maps and linear maps are black-box functions,
the structure of the Jacobian is completely unknown.

**We can't tell which rows and columns are orthogonal without first materializing a Jacobian matrix.**
But if we fully materialize a Jacobian via traditional AD, ASD isn't needed.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparsity_pattern.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/coloring.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 10: The two elementary steps in ASD: (a) sparsity pattern detection, (b) coloring of the sparsity pattern.
</div>

The solution to this problem is shown in Figure 10:
in order to find orthogonal columns (or rows), we don't need to materialize the full Jacobian.
Instead, it is enough to materialize a binary sparsity pattern of the Jacobian.
This pattern contains enough information to color it.

Performance is key: For one-off computations, these two steps need to be faster than the computation of columns or rows they allow us to skip. Otherwise, we didn't gain any performance.
As we will see in later benchmarks, this level of performance can be achieved.
Additionally, if we need to compute Jacobians multiple times and are able to reuse the sparsity pattern, 
the cost of sparsity pattern detection and coloring can be amortized over time.

## Pattern detection

Sparsity pattern detection can be thought of as a binary version of AD.
Mirroring the diversity of existing approaches to AD,
there are also many possible approaches to sparsity pattern detection,
each with their own advantages and tradeoffs.

The method we will present here corresponds to a binary forward-mode AD system 
in which performance is gained by compressing matrix rows.
*TODO: Alternatives include Bayesian probing, ...* 
<!-- TODO: cite a wide list of approaches here -->

### Compressing Jacobians

Our goal with sparsity pattern detection is to quickly materialize the binary pattern of the Jacobian.
One way to achieve better performance than traditional AD is to compress of rows of matrices to index sets.
The $i$-th row of the Jacobian corresponds to 

$$ \big( \Jf \big)_{i,:} 
= \left[\dfdx{i}{j}\right]_{1 \le j \le n}
= \begin{bmatrix}
    \dfdx{i}{1} &
    \ldots      &
    \dfdx{i}{n}
\end{bmatrix} .
$$

This can naively be represented in a computer program by computing and storing using the corresponding $n$ first-order partial derivatives.
However, since we are only interested in the binary pattern 

$$ \left[\dfdx{i}{j} \neq 0\right]_{1 \le j \le n} , $$

we can instead represent the sparsity pattern of the $i$-th column of a Jacobian by the corresponding **index set of non-zero values**

$$ \left\{j \;\Bigg|\; \dfdx{i}{j} \neq 0\right\} . $$

These equivalent sparsity pattern representations are illustrated in Figure 11.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/sparsity_pattern_representations.svg" class="img-fluid" %}
<div class="caption">
    Figure 11: Equivalent sparsity pattern representations: (a) uncompressed matrix, (b) binary pattern, (c) index set (compressed along rows).
</div>

(Since the method we are about to show is essentially a binary forward-mode AD system, we compress along rows.)

### Propagating index sets

Figure 12 shows the traditional forward-AD pass we want to avoid:
propagating a full identity matrix through a linear map would materialize the Jacobian of $f$, 
but also all intermediate linear maps.
As previously discussed, this is not a viable option due to its inefficiency and high memory requirements.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode_naive.svg" class="img-fluid" %}
<div class="caption">
    Figure 12: Materializing a Jacobian forward-mode. 
    Due to high memory requirements for intermediate Jacobians, this approach is inefficient or impossible.  
</div>

Instead, we *seed* an input vector with index sets corresponding to the compressed identity matrix. 
An alternative view on this vector is that it corresponds to the index set representation of the Jacobian of the input, since $\frac{\partial x_i}{\partial x_j} \neq 0$ only holds for $i=j$.

Our goal is to propagate this index set such that we get an output vector of index sets 
that corresponds to the Jacobian sparsity pattern.
This idea is visualized in Figure 13.

{% include figure.html path="assets/img/2025-04-28-sparse-autodiff/forward_mode_sparse.svg" class="img-fluid" %}
<div class="caption">
    Figure 13: Propagating an index set through a linear map to obtain a sparsity pattern.  
</div>

### Alternative evaluation

Instead of going into implementation details,
we want to provide some intuition on the second key ingredient of our forward-mode sparsity detection: 
**alternative function evaluation**.

We will demonstrate this on a second toy example, the function

$$ f(\vx) = x_1 + x_2x_3 + \text{sgn}(x_4) .$$

The corresponding computational graph is shown in Figure 14,
where circular nodes correspond to elementary operators,
in this case addition, multiplication and the sign function.

<!-- TODO: add graph -->

<div class="caption">
    Figure 14: Computational graph of the function $ f(\vx) = x_1 + x_2x_3 + \text{sgn}(x_4) $, annotated with corresponding index sets.  
</div>

As discussed in the previous section,
all inputs are seeded with their respective input index sets.
Figure 14 annotates these index sets on the edges of the computational graph.
Our system for sparsity detection must now perform an **alternative evaluation of our computational graph**.
Instead of computing the original function, 
each operator must correctly propagate and accumulate the index sets of its inputs, 
depending on whether an operator has a non-zero derivative or not.  

Since addition and multiplication globally have non-zero derivatives with respect to both of their inputs, 
the index sets of their inputs are accumulated and propagated. 
The sign function has a zero-valued derivatives for any input value. 
It therefore doesn't propagate the index set of its input. 
Instead, it returns an empty set.

*TODO: switch to multivariate function, quickly discuss resulting Jacobian.*
<!-- TODO -->

<!-- Coloring techniques? -->
### Matrix coloring

Coloring techniques, such as row and column colorings, are used to determine the seed vectors needed to compute the minimal number of directional derivatives required to recover all the non-zero entries of Jacobians.
A seed vector is a linear combination of the rows or columns of the identity matrix.
When a directional derivative of a vector-valued function is taken along this seed vector, the result is a linear combination of the corresponding rows or columns of the Jacobian.

Theses methods relies on reverse or forward automatic differentiation (AD) to efficiently recover the compressed rows or columns of the Jacobian.
This is particularly valuable when dealing with sparse Jacobians, as it enables significant compression without overlap among the non-zero entries.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/smc_column_compressed.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/smc_column_decompressed.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 15: Sparse jacobian compressed by columns (left) and decompressed (right).
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/smc_row_compressed.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/smc_row_decompressed.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 16: Sparse jacobian compressed by rows (left) and decompressed (right).
</div>

When the Jacobian contains at least one dense row and column, unidirectional coloring methods become less efficient.
In such cases, the number of required directional derivatives is as large as the number of rows or columns in the Jacobian.
To address this, bicoloring combines forward and reverse AD to reduce the total number of directional derivatives.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/rectangle_column_coloring.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/rectangle_row_coloring.svg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-sparse-autodiff/rectangle_bicoloring.svg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 17: Row coloring (left), column coloring (center), and bicoloring (right) of a sparse Jacobian containing two dense rows and columns.
</div>

## Second-order sparse differentiation

## Demonstration
