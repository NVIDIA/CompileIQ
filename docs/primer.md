# About Evolutionary Algorithms

This page provides a gentle introduction to Evolutionary Algorithms, in particular to terms used by CompileIQ. It will serve as a useful reference for developers looking into setting up their own searches with CompileIQ.

## Introduction to Evolutionary Algorithms

Evolutionary Algorithms (EAs) are optimization methods inspired by biological evolution. In CompileIQ, they work by:

- Creating an initial population of potential solutions (DNA)
- Evaluating each solution's fitness
- Selecting the best solutions (culling)
- Creating new solutions through mutation
- Repeating this process for multiple generations

The target application’s settings are encoded into `Genes`, where each gene in a DNA can be thought to represent a parameter to tune, command line option, or similar setting. The full collection of genes constructs a single DNA, and the entire collection of DNA is the `Pool` or `population`.

The `search space` is defined by the number and magnitude of genes in a DNA. If, for example, there are two genes in the DNA, each with a range of 1 to 100 and a step of 1, the search space has a size of 10,000 unique locations. Even with this trivial setup this space has already grown enough to be exceedingly difficult to search manually.

Imagine each DNA in the pool as an independent agent searching for an optimal solution in the space defined by the genes. Because the genes are all randomly initialized, each agent effectively starts exploration from a random location in the space. The larger the pool of DNA, the more robust the search. Typically, a pool size should be at least 3x the number of genes in a DNA, roughly at least three unique locations in space to start the search for each gene. However, largest pool size that can be reasonably afforded is just as good.

CompileIQ begins the process by initializing each gene in every DNA to a random value and evaluates the fitness of every DNA in the pool. The quality of the gene pool is improved every generation by the cyclic application of the genetic operations Cull, Mate, and Mutate.

The `cull` and `mating` process is the primary way CompileIQ adjusts the gene values (locations in space) at the end of every generation.

`Culling` sets aside the fittest individuals for mating / cross-over.

During `mating`, two parents are selected at random (from the culled) and produce two offspring. These offspring have a combination of genes from each parent. The offspring occupy locations in space bounded by the hyper-cube with parents at opposite corners. This is done repeatedly until the entire culled population has a chance to reproduce.

The offspring undergo `mutation` and are combined with DNA that live-on to create a new pool. In mutation, randomly selected offspring DNA will have some of the genes altered. Mutations are not bounded as opposed to Mating / Cross-over. Mutation helps avoid local minima / maxima and is also responsible for continuous exploration even during later stages of CompileIQ.

However, too much mutation will degrade Evo’s ability to refine good solutions into better solutions. The whole process is repeated until the pool has dropped below a low limit for genetic diversity or a set number of generations.

`Diversity` is the percentage of uniqueness in the entire pool: At 100% all DNA are unique and at 0% all DNA are identical. The diversity decreases as better performing solutions are found. This is because the successful genes reproduce more often and increase in number in the pool. CompileIQ will change its behavior at various stages based on the current diversity. When the diversity drops below specified thresholds, CompileIQ will slow exploration and focus more on refining good solutions into better ones.

CompileIQ strives to maintain a healthy balance between exploration and refinement throughout the process by using the built in Diversity Manager. It does so adaptively by using a pre-trained model which intelligently navigates CompileIQ through complex search spaces and ensures it does not get stuck at a local optima. This allows users to be able to confidently use CompileIQ for a variety of problems without getting overwhelmed by hyperparameters and instead dedicate more time to optimization problem at hand.
