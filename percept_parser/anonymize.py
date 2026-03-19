import re
import json
import random
from pathlib import Path

import pandas as pd


def recurse(json_data, parent_key=None):
    """Function to recursively iterate through a JSON object"""

    if parent_key is None:
        parent_key = []
    elif parent_key[-1] == 'Groups':
            pass

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            full_key = parent_key + [key]
            yield from recurse(value, full_key)
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            full_key = parent_key + [i]
            yield from recurse(item, full_key)
    else:
        yield parent_key, json_data



def deep_update(full_key, new_value, json_data, verbose=True):
    """Function to update the value for a key deep in a JSON object"""
    if len(full_key) == 1:
        if verbose:
            print(f'Updating {full_key} value: {json_data[full_key[0]]} --> {new_value}')
        json_data[full_key[0]] = new_value
    else:
        deep_update(full_key[1:], new_value, json_data[full_key[0]])

def remove_by_key(key_regex, json_data, new_value="REMOVED", verbose=True):
    """Function to remove a key from a JSON object"""
    for full_key, value in recurse(json_data):
        if isinstance(full_key[-1], str) and key_regex.search(full_key[-1]):
            deep_update(full_key, new_value, json_data, verbose=verbose)

def remove_serial(json_data, verbose=True):
    """Function to search for and remove device serial numbers"""
    if verbose:
        print(f'Removing device serial info')
    serial_re = re.compile(r'[sS]erial[nN]umber')
    remove_by_key(serial_re, json_data, new_value="##########", verbose=verbose)


def remove_patient_info(json_data, verbose=True):
    """Function to search for and remove a patient name"""
    if verbose:
        print(f'Removing patient personal info')

    name_re = re.compile(r'[pP]atient.*[nN]ame')
    remove_by_key(name_re, json_data, verbose=verbose)

    dob_re = re.compile(r'[pP]atient.*[bB]irth')
    remove_by_key(dob_re, json_data, verbose=verbose)


def find_and_replace_values(json_data, search_key, replace_func, verbose=True):
    """Function to search for and replace values in a JSON object"""
    search = re.compile(search_key)
    for full_key, value in recurse(json_data):
        if isinstance(value, str) and ".000Z" in value:
            pass
        if isinstance(value, str) and search.search(value):
            deep_update(full_key, replace_func(value), json_data, verbose=verbose)

def obfuscate_dates(json_data, verbose=True, shift=None):
    """
    Function to obfuscate all dates by adding a random shift
    We need to run twice since there are two different date formats used in the file
    """
    date_shift = pd.Timedelta(seconds=random.randint(-10**8, 10**8)) if shift is None else shift

    if verbose:
        print(f'Obfuscating metadata dates')
    # This is the format used in the metadata fields
    re_key = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z'
    shift_func = lambda dt: (pd.to_datetime(dt) + date_shift).strftime('%Y-%m-%dT%H:%M:%SZ')
    find_and_replace_values(json_data, re_key, shift_func, verbose=verbose)

    if verbose:
        print(f'Obfuscating data timestamps')
    # This is the format used in the actual data timestamps
    re_key = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z'
    def shift_func(dt):
        new_dt = pd.to_datetime(dt) + date_shift
        new_str = new_dt.strftime(f'%Y-%m-%dT%H:%M:%S.{int(new_dt.microsecond/10**3):03d}Z')
        return new_str
    find_and_replace_values(json_data, re_key, shift_func, verbose=verbose)

    return date_shift


def anonymize(json_filepath, output=None, out_dir=None, serial=True, name=True, dates=True, verbose=True, shift=None):
    """Function to anonymize a JSON object"""
    with open(json_filepath, 'r') as file:
        json_data = json.load(file)

    if serial:
        remove_serial(json_data, verbose=verbose)
    if name:
        remove_patient_info(json_data, verbose=verbose)
    if dates:
        if verbose:
            print(f'Obfuscating all dates')
        shift_used = obfuscate_dates(json_data, verbose=verbose, shift=shift)

    if output is None:
        json_filepath = Path(json_filepath)
        filename = json_filepath.stem
        str_date = re.search(r'\d{8}T\d{6}', filename)
        dest_dir = Path(out_dir) if out_dir is not None else json_filepath.parent
        if dates and str_date:
            new_date = pd.Timestamp(str_date.group()) + shift_used
            new_date_str = new_date.strftime('%Y%m%dT%H%M%S')
            if verbose:
                print(f'Shifting date in filename: {str_date.group()} --> {new_date_str}')
            filename = f'Report_Json_Session_Report_{new_date_str}_anonymized.json'
            output = dest_dir / filename
        else:
            output = dest_dir / (json_filepath.stem + '_anonymized.json')
    else:
        output = Path(output)
    with open(output, 'w') as file:
        if verbose:
            print(f'Writing anonymized JSON to {output}')
        json.dump(json_data, file, indent=4)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("json_filepath", help="Path to the JSON file to anonymize")
    parser.add_argument("--output", help="Path to save the anonymized JSON file")
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
    parser.add_argument(
        "--skip-dates",
        help="Skip obfuscating dates. This may make PHI removal incomplete!",
        action="store_true")
    parser.add_argument(
        "--time-shift", type=int,
        help="Manually selected timeshift in seconds. This will override the automatic random timeshift"
    )

    args = parser.parse_args()
    manual_shift = pd.Timedelta(seconds=args.time_shift) if args.time_shift else None
    anonymize(args.json_filepath, output=args.output, verbose=args.verbose, shift=manual_shift)