from typing import Any, Dict, overload
import base64
from compileiq.search_spaces import base as ss
from compileiq.search_spaces.models import ParamConfig


def _encode_for_core(value: str) -> str:
    """
    Encodes a string to be used in Core to avoid issues with special characters.
    """
    # We need to add a prefix in case the hex is just a number.
    # Core will not accept number-only values as valid identifiers.
    return base64.b64encode(value.encode()).decode()


def _decode_from_core(string: str) -> str:
    """
    Decodes base64 string back to its original form.
    """
    return base64.b64decode(string).decode()


def flatten_nested_dict(dna_search_space: Dict[str, Any]) -> Dict[str, ParamConfig]:
    """
    Converts a dictionary with nested keys into a flat dictionary. We encode the keys
    as a way to nest them back again without losing the original structure, independent
    of user-defined names.

    Args:
        dna_search_space (Dict[str, Dict]): The nested search space dictionary.

    Returns:
        A flat dictionary with encoded string keys in hex format joined
        by underscores.

    """
    flat_dict = {}

    def flatten_dict(dict_to_flat: Dict, parent_key: str = ""):
        for key, val in dict_to_flat.items():
            key = _encode_for_core(key)
            new_key = f"{parent_key}_{key}" if parent_key else key

            if isinstance(val, dict):
                flatten_dict(val, new_key)
            else:
                flat_dict[new_key] = val

    flatten_dict(dna_search_space)

    return flat_dict


def restore_nested_search_space(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Restore the nested dictionary structure provided by the user at search space.
    We are reversing the processes of `flatten_nested_dict`.

    Args:
        params (Dict[str, str]): The flat dictionary with encoded keys.

    Returns:
        A nested dictionary structure that matches the original search space.
    """

    restored: dict[str, object] = {}
    for key, val in params.items():
        # Decoding the hex-encoded key and splitting it into parents
        hierarchy = list(map(_decode_from_core, key.split("_")))
        # Restore the nested dictionary structure
        last_key = hierarchy.pop(-1)
        nested: dict[str, object] = {last_key: val}

        # Rebuild the hierarchy in reverse order
        for nest_key in reversed(hierarchy):
            nested = {nest_key: nested}

        # Merge the nested structure into the restored dictionary
        restored = _merge_nested_dictionaries(restored, nested)

    return restored


def _merge_nested_dictionaries(dict1: Dict, dict2: Dict) -> Dict:
    """
    Given two nested dictionaries, merges them together. They will be nested
    under the matched keys without overwrites, for example:
    val1 ={'y': {'yy1': {'yyy1': {'yyyy1': 2}}}}
    val2 = {'y': {'yy1': {'yyy2': 2}}}

    Output: {'y': {'yy1': {'yyy1': {'yyyy1': 2}, 'yyy2': 2}}}
    """
    # Warning: dict1 is updated by reference, so it will be modified
    for key in dict2:
        # if keys exists in both dictionaries, we need to look
        # deeper into the nesting for the merge
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            _merge_nested_dictionaries(dict1[key], dict2[key])
        else:
            # if key does not exist, just dump entire content
            dict1[key] = dict2[key]

    return dict1


@overload
def _literal_dive(val: dict, knockout: float | None = None) -> dict: ...


@overload
def _literal_dive(val: str | int | float, knockout: float | None = None) -> ParamConfig: ...


def _literal_dive(
    val: dict | str | int | float,
    knockout: float | None = None,
) -> ParamConfig | dict:
    """
    Dives down into nested structure until it finds leaves, then converts them
    into literal search space elements.
    """
    if isinstance(val, dict):
        return {k: _literal_dive(v, knockout) for k, v in val.items()}
    else:
        return ss.literal(const_value=val, knockout_prob=knockout)


def save_compiler_config(file_path: str, binary_blob: str):
    """
    Saves a binary blob to a file.

    Args:
        file_path: The path where the file will be saved.
        binary_blob: The binary data to be written to the file.
    """
    with open(file_path, "wb") as f:
        f.write(bytes.fromhex(binary_blob))


def load_compiler_config(file_path: str) -> str:
    """
    Load a binary blob to a hex string

    Args:
        file_path: The path to load the file from.

    Returns:
        File content as a hex encoded string.
    """
    with open(file_path, "rb") as f:
        return f.read().hex()
