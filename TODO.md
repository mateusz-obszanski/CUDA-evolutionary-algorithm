# TODO

- DeviceArr::print add max n elements parameter

## EA Coding

- Open Travelling Salesman Problem - NP-complete, can be extended to Closed TSP
  (permute only cities that are not source and destination, add those at the end stage)

- Tensor dimensions (options)

  1. 1 tensor 3D

     1. #Population
     1. #Individual
     1. #Gene

  1. N populations - N tensors (2D):

     1. #Individual
     1. #Gene

        Can be computed using CUDA streams  
         Difficulty for migration operator  
         Slightly easier for everything else (mutation, crossover etc.)

## To implement

- [] **solution encoding**
- [] **random initialization (cuRAND?) - random permutation**
- [] **Mutation operator**
  - [] **random permutation of k elements**
  - [] ? permutation of sections (different lengths or not?)
- [] **Cross-over operator**
  - [] **random gene selection from one parent and complement from second parent**
  - [] swapping chromosome endings
  - [] ? swapping chromosome sections
- [] **Fix operator**
  - [] **fixing duplicates after cross-over**
    - [] **finding k duplicates**
    - [] **random k - 1 duplicate selection**
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
  - [] **Selection of migrants**
    - [] **Fixed number of migrants** / random number of migrants (equal size in each round)
    - [] **Strategy**: **`k`-best**/random choice/tournament
- [] **Loss function calculation**
- [] **solution decoding**
- [] **Learning process**
  - [] **Zip everything together**
  - [] **Stop after n iterations**
  - [] **Saving currently best solution**
  - [] ? early stop?
  - [] ? epochs/checkpoints and adaptive learning parameters
