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
    search_space_config_filepath = os.path.join(folder, SEARCH_SPACE_CONFIG_FILENAME)

    if sys.platform == "win32":
        search_space_config_filepath = search_space_config_filepath.replace("\\", "/")
        main_config_filepath = main_config_filepath.replace("\\", "/")

    return main_config_filepath, search_space_config_filepath


def setup_legacy_search_config(
    search_configuration: InternalSearchConfiguration,
    main_config_filepath: str,
):
    # Creating main_config.json in `folder`
    config_dict = search_configuration.to_json_dict()

    with open(main_config_filepath, "w") as fp:
        json.dump(config_dict, fp, indent=2)


def setup_search_space(
    search_space_input: (
        Dict[str, Any] | pathlib.Path | List[Dict | pathlib.Path] | SearchSpaceProvider
    ),
    search_space_config_filepath: str,
) -> str | List[str]:
    """
    CompileIQ's core expects a search-space config file referenced by main_config.json.
    This function converts a dictionary-style config into JSON and copies it into `folder`.
    File-backed search spaces are copied into the run cache as-is.
    """

    entries: List[Dict[str, Any] | pathlib.Path | SearchSpaceProvider]
    if isinstance(search_space_input, list):
        entries = list(search_space_input)
    else:
        entries = [search_space_input]

    search_files = []
    base_path = pathlib.Path(search_space_config_filepath)

    # Set up one core search-space file per input.
    for i, search_space in enumerate(entries):
        # Renaming in case we have multiple configs
        if len(entries) > 1:
            current_path = base_path.with_name(f"{i}_{base_path.name}")
        else:
            current_path = base_path

        if isinstance(search_space, dict):
            flat_search_space = flatten_nested_dict(search_space)

            # Convert dictionary search spaces into the core JSON format.
            search_space_json = _setup_search_space_with_dict(flat_search_space)
            with open(current_path, "w") as fp:
                fp.write(search_space_json)

        elif isinstance(search_space, pathlib.Path) and search_space.exists():
            # File-backed search spaces are already in a core-readable format.
            shutil.copy(search_space, current_path)

        elif isinstance(search_space, pathlib.Path) and not search_space.exists():
            raise FileNotFoundError(f"Search-space config file not found: {search_space}")
        else:
            raise ValueError("CompileIQ Search Spaces need to be of type dict or path to a file")

        search_files.append(str(current_path))

    return search_files if len(search_files) > 1 else search_files[0]


def _setup_search_space_with_dict(search_space_dict: Mapping[str, ParamConfig]) -> str:
    """
    Creates a JSON string representing the search-space configuration. The JSON
    follows the core search-space schema and is parsed by core.
    """
    search_space_list = ["{"] + list(search_space_dict.keys()) + ["}"]
    model = SearchSpaceFileModel(
        classes=dict(search_space_dict), parameter_layout=search_space_list
    )
    return model.model_dump_json(exclude_none=True, indent=2, by_alias=True)
