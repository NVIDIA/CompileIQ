from compileiq.ciq import Search
from compileiq.types import SearchConfiguration
import os
from loguru import logger

CONFIG_FOLDER = "./data/"


def main():
    main_config = SearchConfiguration(
        pool_size=12,
        generations=3,
        mutate_rate=0.5,
        problem_type="min",
        num_objectives=1,
    )

    for config in os.listdir(CONFIG_FOLDER):
        logger.info(f"Processing config file: {config}")
        dna_config = os.path.join(CONFIG_FOLDER, config)

        try:
            tuner = Search(
                objective_function=lambda _: None,
                search_space=dna_config,
                search_config=main_config,
            )
            tuner.sample(32)
            # Validating we can extract samples from compiler search spaces
            logger.success(f"Correctly processed {config}")
        except Exception as e:
            if "invalid" in config:
                logger.success(f"Correctly failed processing {config}")
                continue
            else:
                raise RuntimeError(f"Failed processing {config}, but should have passed.") from e


if __name__ == "__main__":
    main()
