"""(Optional) intermediary to collect all relevant GUI materials in one module.
Ideally, this would avoid confusion with separate imports and accessing `interface` vs `processing`.
The user only wants to utilise interface, but may use processing elements separate from this.

The user interface can be launched using the interface.create() function.
If importing to Jupyter Notebook, this function is accessible in the solver namespace and the GUI object can be stored.
Otherwise, this script allows the program to be launched automatically by executing solver.py with python on command line.
"""

from interface import GUI, create
# from processing import *


if __name__ == "__main__":
    create()