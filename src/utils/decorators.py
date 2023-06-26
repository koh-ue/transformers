#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .printcolor import print_in_bold_green

def show_start_end(func):
    def wrapper(*args, **kwargs):
        function_name = func.__name__
        print_in_bold_green(f"start {function_name} ...")
        res = func(*args, **kwargs)
        print_in_bold_green(f"end {function_name} ...")
        return res
    return wrapper