# Getting Started with CompileIQ

This guide walks you through creating your first evolutionary search using CompileIQ. We'll create a simple optimization problem that demonstrates the core concepts.

## Prerequisites

Before starting, make sure you have CompileIQ installed. For detailed installation instructions, see the [Installation Guide](install.md).

## Basic Example: Single Objective Optimization

In this example, we'll optimize a simple mathematical function with multiple parameters, showcasing different types of search spaces.

### Create Your First Search

```python
from compileiq.ciq import Search
from compileiq.types import SearchConfiguration
import compileiq.search_spaces.base as ss

# Define the objective function to minimize
def objective(config):
    score = config["x"] ** 2 + config["y"]
    return score

# Define the search space
dna_config = {
    "x": ss.range(start=1.0, end=20.0, step=0.5),  # Continuous range
    "y": ss.choice([1, 2, 3]),  # Discrete choices
    "z": ss.literal(const_value="constant", knockable_threshold=0.5),  # Binary parameter
}

# Configure the evolutionary search
main_config = SearchConfiguration(
    pool_size=12,      # Size of population
    generations=3,      # Number of iterations
    mutate_rate=0.5,   # Mutation probability
    problem_type="min", # Minimize objective
)

# Create and run the search
tuner = Search.single_objective(
    objective_function=objective,
    search_space=dna_config,
    search_config=main_config,
)

results = tuner.start()
print(f"Best Result: {results.get_best_result()}")
```

### Understanding the Code

1. **Objective Function**
    - Takes a configuration dictionary as input
    - Returns a score to optimize (minimize in this case)
    - Can include any computation you need to evaluate solutions

2. **Search Space Definition**
    - `range`: Continuous numerical range with step size
    - `choice`: List of discrete options
    - `literal`: Binary parameter that can be knocked out

3. **Evolution Configuration**
    - `pool_size`: Number of solutions in each generation
    - `generations`: Number of iterations
    - `mutate_rate`: Probability of mutation
    - `problem_type`: "min" or "max"

4. **Running the Search**
    - Create a `Search` instance with your configurations
    - Call `start()` to run the optimization
    - Access results through the returned Results object

### Multi-Objective Searches

CompileIQ also supports multi-objective searches. Let's modify the previous example to now return two different results:

```python
from compileiq.ciq import Search
from compileiq.types import SearchConfiguration
import compileiq.search_spaces.base as ss

def multiobjective(config):
    score_1 = config["x"] ** 2 + config["y"]
    score_2 = config["y"] ** 2 + config["x"]
    return score_1, score_2 # returning two scores


dna_config = {
    "x": ss.range(start=1.0, end=20.0, step=0.5),  # Continuous range
    "y": ss.choice([1, 2, 3]),  # Discrete choices
    "z": ss.literal(const_value="constant", knockable_threshold=0.5),  # Binary parameter
}

main_config = SearchConfiguration(
    pool_size=12,
    generations=3,
    mutate_rate=0.5,
    problem_type="min",
)

tuner = Search.multi_objective(
    objective_function=multiobjective,
    search_space=dna_config,
    search_config=main_config,
    num_objectives=2,
)

results = tuner.start()
print(results.pareto_front())
```

The two important differences are calling `Search.multi_objective(...)` with `num_objectives=2` and returning two values from the objective function.

The return value of `results.pareto_front()` is a list with the Pareto-efficient rows.

### Sample the Search Space

You can use `Search.sample` to retrieve randomly generated samples from your search space. This helps you understand what will be forwarded to your objective function.

```python
# Create and run the search
tuner = Search.single_objective(
    objective_function=objective,
    search_space=dna_config,
    search_config=main_config,
)

results = tuner.sample(5)
```

In this example, we retrieve 5 samples, and you should get a list with 5 sampled values.

```python
[{'x': 4.0, 'y': 2}, {'x': 17.5, 'y': 1}, {'x': 5.5, 'y': 3, 'z': 'constant'}, {'x': 15.0, 'y': 2, 'z': 'constant'}, {'x': 11.5, 'y': 3, 'z': 'constant'}]
```

## Next Steps

CompileIQ supports more advanced features:

- Multi-objective optimization
- Distributed computing
- Checkpointing for long-running searches
- Custom normalization strategies

## References

- [Examples Gallery](https://github.com/NVIDIA/CompileIQ/blob/main/examples) - More quick examples including:
  - Multi-objective optimization (`multi_objective.py`)
  - Distributed searches (`distributed.py`)
  - Checkpointing (`checkpointing.py`)
  - XGBoost integration (`ciq_xgboost.py`)
