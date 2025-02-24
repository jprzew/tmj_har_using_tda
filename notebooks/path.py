#!/usr/bin/env python3

import sys
import os

module_path = os.path.abspath(os.pardir)
module_path = os.path.join(module_path, 'src')
if module_path not in sys.path:
    sys.path.append(module_path)
