import json
import pathlib
import os
import sys
import shutil
import warnings
from typing import Dict, Any, List, Mapping
from compileiq.config.const import (
    MAIN_CONFIG_FILENAME,
    SEARCH_SPACE_CONFIG_FILENAME,
    _CACHE_DIR,
)
from compileiq.types import InternalSearchConfiguration
from compileiq.utils.helpers import flatten_nested_dict
from compileiq.search_spaces.compilers import SearchSpaceProvider
from compileiq.search_spaces.models import ParamConfig, SearchSpaceFileModel


def clear_cache():
    """
    Warning: This will remove the entire .cache/compileiq data
    """
    warnings.warn(
        "This function removes the entire .cache folder for compileiq. "
        "You may lose other experiments data or break running experiments.",
        stacklevel=2,
    )
    if pathlib.Path(_CACHE_DIR).exists():
        shutil.rmtree(_CACHE_DIR)


def get_core_filepaths(folder: str | os.PathLike) -> tuple[str, str]:
    """
    All filenames where core reads or writes from/to
    """
    main_config_filepath = os.path.join(folder, MAIN_CONFIG_FILENAME)
    dna_config_filepath = os.path.join(folder, SEARCH_SPACE_CONFIG_FILENAME)

    if sys.platform == "win32":
        dna_config_filepath = dna_config_filepath.replace("\\", "/")
        main_config_filepath = main_config_filepath.replace("\\", "/")

    return main_config_filepath, dna_config_filepath


def setup_legacy_search_config(
    search_configuration: InternalSearchConfiguration,
    main_config_filepath: str,
):
    # Creating main_config.json in `folder`
    config_dict = search_configuration.to_json_dict()

    with open(main_config_filepath, "w") as fp:
        json.dump(config_dict, fp, indent=2)


def setup_search_space(
    dna_search_space: (
        Dict[str, Any] | pathlib.Path | List[Dict | pathlib.Path] | SearchSpaceProvider
    ),
    dna_config_filepath: str,
) -> str | List[str]:
    """
    CompileIQ's Core expects a dna config file following the main.config.
    This function converts a dictionary-style config into JSON and copies into `folder`.
    Legacy .config files (S-expression format) are copied as-is for backward compatibility.
    """

    entries: List[Dict[str, Any] | pathlib.Path | SearchSpaceProvider]
    if isinstance(dna_search_space, list):
        entries = list(dna_search_space)
    else:
        entries = [dna_search_space]

    search_files = []
    base_path = pathlib.Path(dna_config_filepath)

    # Setting up dna.config files
    for i, search_space in enumerate(entries):
        # Renaming in case we have multiple configs
        if len(entries) > 1:
            current_path = base_path.with_name(f"{i}_{base_path.name}")
        else:
            current_path = base_path

        if isinstance(search_space, dict):
            flat_search_space = flatten_nested_dict(search_space)

            # Converting Dictionary DNA into JSON format
            search_space_json = _setup_dna_with_dict(flat_search_space)
            with open(current_path, "w") as fp:
                fp.write(search_space_json)

        elif isinstance(search_space, pathlib.Path) and search_space.exists():
            # Expects a lisp-like format used on Legacy CompileIQ
            # Copying the dna to the cache folder for usage
            shutil.copy(search_space, current_path)

        elif isinstance(search_space, pathlib.Path) and not search_space.exists():
            raise FileNotFoundError(f"Dna config file not found: {search_space}")
        else:
            raise ValueError("CompileIQ Search Spaces need to be of type dict or path to a file")

        search_files.append(str(current_path))

    return search_files if len(search_files) > 1 else search_files[0]


def _setup_dna_with_dict(dna_dict: Mapping[str, ParamConfig]) -> str:
    """
    Creates a JSON string representing the DNA configuration. The JSON
    follows the core search-space schema and is parsed by core.
    """
    search_space_list = ["{"] + list(dna_dict.keys()) + ["}"]
    model = SearchSpaceFileModel(classes=dict(dna_dict), dna=search_space_list)
    return model.model_dump_json(exclude_none=True, indent=2, by_alias=True)
