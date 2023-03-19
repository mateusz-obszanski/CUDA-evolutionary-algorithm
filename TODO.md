# TODO

Use thrust/random.h whenever possible  
thrust/random.h examples:

- [https://github.com/NVIDIA/thrust#examples]
- [https://stackoverflow.com/a/12614606]

**Thrust [docs](https://nvidia.github.io/thrust/)**

## Misc

Thrust - maybe use cudaMalloc2D/cudaFree wrapped in RAII object
and pass it to thrust algorithms for better memory layout.

## EA Coding

- Open Travelling Salesman Problem - NP-complete, can be extended to Closed TSP
  (permute only cities that are not source and destination, add those at the end stage)

- Tensor dimensions  
  options:

  1. 1 tensor 3D

     1. #Population
     2. #Individual
     3. #Gene

     :thumbsup::

     - everything copied at once, device in-memory migration possible

     :thumbsdown::

     - memory limits
     - grid size limits

  2. **N populations - N tensors (2D):**

     1. #Individual
     2. #Gene

        Can be computed using CUDA streams  
         Difficulty for migration operator  
         Slightly easier for everything else (mutation, crossover etc.)

     :thumbsup::

     - easy
     - memory restricts less
     - still option for migration of chromosomes in device memory
       if all arrays fit in GPU memory - copying between GPU arrays
     - possible (and maybe default) migration in CPU RAM

  3. whole population inside a single CUDA block, thread === one chromosome

     :thumbsup::

     - blazingly fast shared memory

     :thumbsdown::

     - strong hardware limitations
       - max 1024 chromosomes in population
       - #chromosomes $\times$ #genes limited by shared memory
       - 48kB (check GPU spec) of shared memory, limited parallel block execution on a single Streaming Multiprocessor
     - no DRAM burst due to thread-memory mapping. Or at least complicated implementation and reusing threads in different conventions inside a single kernel or between them.
     - CUDA thrust library - impossible/very difficult to meaningfully use its algorithms

## To implement

- [] [**solution encoding**](#solution-encoding)
- [] [**solution decoding**](#solution-decoding)
- [] [**Loss function calculation**](#loss-function)
- [] **random initialization** (**thrust**(preferably)/cuRAND) - random permutation
- [] [**Mutation operator**](#mutation)
  - [] [**random permutation of $k$ elements**](#elementwise-permutation)
  - [] ? [permutation of sections](#permutation-of-sections)
    (different lengths or not?)
- [] [**Cross-over operator**](#crossover)
  - [] [**mask-complement swap**](#mask-complement-swap)
  - [] [swapping chromosome endings](#end-swap)
  - [] ? swapping chromosome sections
- [] [**Fix operator**](#fixing-chromosomes)
  - [] **fixing duplicates after cross-over**
    - [] **finding $k$ duplicates**
    - [] **random $k - 1$ duplicate selection**
    - [] **filling with random permutation of unused cities**
- [] **Selection operator**
  - [] **pairs of consequtive best candidates**
  - [] ? totally random selection
  - [] select parents with probability proportional to their loss function values
  - [] tournament selection
- [] Rivalization/natural selection of offpring
  produce more offspring than needed and let them fight? Probably equivalent
  to selection of parents
- [] **Migration operator**
  <!-- TODO migration type - what options are even available? see literature -->
  Think about migration type
  - [] **Selection of migrants**
    - [] **Fixed number of migrants** / random number of migrants (equal size in each round)
    - [] **Strategy**: **$k$-best**/random choice/tournament
- [] **Learning process**
  - [] **Assemble everything together**
  - [] **Stop after n iterations**
  - [] **Saving currently best solution**  
     Options:
    - after each generation
      - :thumbsup: will remember THE BEST solution
      - :thumbsdown: will require some GPU memory to remember the
        solution OR copying to host memory (synchronization :weary:)
    - after each epoch (epoch === migration)
      - :thumbsup: no synchronization, less memory overhead (still, would not take much memory to save one chromosome)
      - :thumbsdown: will forget the really best solution
  - [] ? early stop?
  - [] ? epochs/checkpoints and adaptive learning parameters

## Solution encoding

- OTSP  
  If start and/or stop indices are fixed, do not include them in the genom
- TSP  
  If start-stop index is fixed, remove it from the genom

## Solution decoding

- OTSP  
  If start (and/or stop) index is fixed, prepend (append) it to the solution
- TSP  
  If start-stop index is fixed, prepend and append it to the solution

## Loss function

- DEBUG: distance metric from some permutation vector
- target: OTSP (for benchmarking)

## Mutation

### Elementwise permutation

#### $k$-permutation

draw $k$ indices

- needs $k$ threads
- select $k$ indices - options:
  - pass from CPU?
    - :thumbsup: easiest solution, choice without replacement
    - :thumbsup: can be parallel to kernel execution
    - :thumbsdown: requires device-host pre/post synchronization
  - `kShuffle`
- copy them to shared memory $a_1$ - read indices
- copy to second shared memory array $a_2$ - write indices
- shuffle second array
- swap based on read/write index arrays

shared memory elements: $2 \cdot k$

Kernels/device functions:

- `kShuffle`

  - `chooseWithReplacement`
    - `randint` - normal distribution
  - Based on above - `chooseWithoutReplacement`

    - $k$ kernels choose indices (`atomicWrite`). If $i$ are
      duplicated, use all $k$ threads to generate remaining ones.
      Repeat as long as needed. Then, `__syncthreads`.

      It can be also used as `discardWithoutReplacement` by filtering
      out the results. If $k > floor(n / 2)$, launch `discardWithoutReplacement`, because then the synchronous part will be shorter - it is better to filter out.

  - selection shuffle (thrust)

  Alternative, simpler choice method: `maskedShuffle`, where mask is probabilistic,
  $p = k/n$, where $k$ - expected number of chosen elements, $n$ - length of chromosome

#### $p$-permutation

probabilistic index choice

- host: select $p_0$
- thread: draw $p$ from uniform distribution
- save index to shared memory if $p < p_0$ (read-indices)
- copy shared array (write-indices)
- shuffle first array
- swap

shared memory elements: $2 \cdot$ (either #genes or maybe expected
value of drawn indices + standard deviation? - this bounds
#shuffle-indices)

### Permutation of sections

kernels/device functions:

- `chooseWithoutReplacement` - to generate array split indices
- `shuffledSequence` - generate shuffled indices of sections
  - `sequence` - thrust?
  - `shuffle`
- `copy(begin, end)`

## Crossover

### Mask-complement swap

kernels/device functions:

- `randMask`
- `maskSwap` - thrust zip with mask, maybe custom `maskSwap` kernel

### End-swap

#### Deterministic

Array even binary split or $k$-binary split into ($[0, k)$ and $[k, end)$)

#### Random binary split

kernels/device functions:

- `randBinarySplitArr`
  - `randInteger<IntegerType>` - generate split index. Split drawn from
    range $[0, n - 2]$. Uniform distribution or normal with some index
    $i$ chosen to be at mean value. $n$ is genome length.
  - deterministic `binarySplitArrAfter(unsigned int i)` - splits before `i`

## Fixing chromosomes

kernels/device functions:

- `boundedCounter` - takes $n$-element output array (already zeroed
  out?) and increments (`atomicInc`) element at index $i$ for each
  $i$ found in input array. Assumption - input array consists of
  numbers in range $[0, n - 1)$.
- `thrust::partition` based on predicate `> 1` with stencil. (first, last) is a sequence $0, 1, ..., n-1$ - indices that will be reordered.
  The stencil - chromosome.
- random permutation (see above) OR plain in-order insertion of unique
  genes at indices of found duplications (or $n$-plications)
