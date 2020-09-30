"""Helper module for common used methods"""
import os
import re


def is_non_zero_file(self, fpath):
    return (os.path.isfile(fpath) and os.path.getsize(fpath) > 0)


def sort_alphanumeric(data):
    def convert = lambda text: int(text) if text.isdigit() else text.lower()

    def alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',
                                    key)]
    return sorted(data, key=alphanum_key)
