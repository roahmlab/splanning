import torch
import os


def get_default_with_warning(base, name, default_val=None):
    if name not in base:
        print(f"Warning: Failed to find {name} in configuration. Using default value ({default_val})")
        return default_val
    return base.get(name)


def validate_get_dtype_device(dtype: str | None, device: str | None):
    '''Validate the data type and device, returning the data type and device to use.

    Args:
        dtype (str | None): The data type to use as a string
        device (str | None): The device to use as a string

    Returns:
        tuple: The torch data type and device to use
    
    Raises:
        ValueError: If the dtype is invalid
    '''
    # validate dtype and device
    if dtype is not None:
        try:
            dtype = getattr(torch, dtype)
        except AttributeError:
            dtype = None
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid dtype specified: {dtype}")
    dtype_device = torch.empty(0, device=device, dtype=dtype)
    device = dtype_device.device
    dtype = dtype_device.dtype
    return dtype, device


def resolve_incomplete_filename(path, resolve_folder=False):
    ''' Simple utility to resolve an incomplete filename or foldername.

    Only searches within the provided folder
    
    Args:
        path (str): The path that may or may not be complete
        resolve_folder (bool): Whether we want to resolve to an incomplete foldername or incomplete filename
    
    Returns:
        resolved_path (str): If a single option is resolvable, the resolved file or folder name found
    
    Raises:
        ValueError: If there are multiple candidate folders or filenames found.
        FileNotFoundError: If there is no matching folder found.
    '''
    # If already complete, just return
    if os.path.exists(path) or (resolve_folder and os.path.isdir(path)) or (not resolve_folder and os.path.isfile(path)):
        return path
    
    # Helper for search
    validation_func = (lambda x: x.is_dir()) if resolve_folder else (lambda x: x.is_file())
    if os.path.exists(path):
        parent_folder = path
        partial_name = ''
    else:
        parent_folder = os.path.dirname(path)
        partial_name = os.path.basename(path)

    matches = [f.name for f in os.scandir(parent_folder) if validation_func(f) and f.name.startswith(partial_name)]
    if len(matches) == 1:
        resolved_path = os.path.join(parent_folder, matches[0])
    elif len(matches) > 1:
        raise ValueError("Error: Multiple matching folders found. Please specify the folder more clearly.")
    else:
        raise FileNotFoundError("Error: No matching folder found.")
    return resolved_path
