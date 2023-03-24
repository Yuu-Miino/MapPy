def get_version_number(st):
    import re

    """
    A function that extracts the version number from a string that is enclosed in single or double quotes.
    by ChatGPT

    Parameters
    ----------
    st : str
        The input string containing the version number enclosed in quotes.

    Returns
    -------
    str or None
        The extracted version number as a string. If no version number is found, returns None.

    Examples
    --------
    >>> get_version_number("version = '0.0.4'")
    '0.0.4'
    >>> get_version_number('version = "0.0.5"')
    '0.0.5'
    """
    pattern = r"(?<=['\"])[^'\"]+(?=['\"])"  # Regular expression pattern
    match = re.search(pattern, st)  # Get the match for the regular expression in the string
    if match is None:
        return None  # Return None if no match is found
    return match.group()  # Return the matched substring
