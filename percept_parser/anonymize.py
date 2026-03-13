import re
import json
import random

import pandas as pd



def recurse(json_data, parent_key=None):
    """Function to recursively iterate through a JSON object"""
    if parent_key is None:
        parent_key = []
    for key, value in json_data.items():
        full_key = parent_key + [key]
        if isinstance(value, dict):
            yield from recurse(value, full_key)
        else:
            yield full_key, value

def deep_update(full_key, new_value, json_data, verbose=True):
    """Function to update the value for a key deep in a JSON object"""
    if len(full_key) == 1:
        if verbose:
            print(f'Updating {full_key} value: {json_data[full_key[0]]} --> {new_value}')
        json_data[full_key[0]] = new_value
    else:
        deep_update(full_key[1:], new_value, json_data[full_key[0]])


def remove_serial(json_data):
    """Function to search for and remove device serial numbers"""
    for full_key, value in recurse(json_data):
        if 'serialnumber' in full_key[-1].lower():
            deep_update(full_key, "##########", json_data)


def remove_patient_info(json_data):
    """Function to search for and remove a patient name"""
    name_re = re.compile(r'[pP]atient.*[nN]ame')
    for full_key, value in recurse(json_data):
        if name_re.match(full_key[-1]):
            deep_update(full_key, "REMOVED", json_data)


def obfuscate_dates(json_data):
    """Function to obfuscate all other dates by adding a random shift"""
    date_shift = pd.Timedelta(seconds=random.randint(-10**8, 10**8))
    date_search = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z')
    for full_key, value in recurse(json_data):
        if isinstance(value, str) and date_search.search(value):
            new_datetime = (pd.to_datetime(value) + date_shift).strftime('%Y-%m-%dT%H:%M:%SZ')
            deep_update(full_key, new_datetime, json_data)


def anonymize(json_filepath, output=None, serial=True, name=True, dates=True):
    """Function to anonymize a JSON object"""
    with open(json_filepath, 'r') as file:
        json_data = json.load(file)

    if serial:
        remove_serial(json_data)
    if name:
        remove_patient_info(json_data)
    if dates:
        obfuscate_dates(json_data)

    output = json_filepath if output is None else output
    with open(output, 'w') as file:
        json.dump(json_data, file, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("json_filepath", help="Path to the JSON file to anonymize")
    parser.add_argument("--output", help="Path to save the anonymized JSON file")

    args = parser.parse_args()

    anonymize(args.json_filepath, output=args.output)