#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . N A M E L I S T
#  read Fortran namelist using f90nml
#  (accounting for unquoted strings of the form %NAME%, which are popular in bacy but not compatible with f90nml)
#
#  2019.8 L.Scheck

import f90nml, re

def read_namelist( fname ) :

    nml_string = ''
    with open( fname, 'r' ) as f :
        for line in  f :
            if '%' in line : # enclose unquoted strings of the form %NAME% with quotes
                nml_string += re.sub( r'([^\'\"])\%([^\%]+?)\%([^\'\"])', r"'%\2%'", line )
            else :
                nml_string += line

    parser = f90nml.Parser()
    nml = parser.reads(nml_string)

    return nml

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    import sys

    print read_namelist(sys.argv[1])
