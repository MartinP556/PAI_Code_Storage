#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
#
# K E N D A P Y . B I N P L O T
#
# 2014-6 L.Scheck

from __future__ import absolute_import, division, print_function
from numpy import *
from matplotlib import pyplot as plt

def binplot( x, y, style='-', color=None, linewidth=1, semilogy=False, semilogx=False, label=None, ax=None ) :
    """plot binned data"""

    if color == None :
        color = 'k'

    if x.size == y.size + 1 :
        X = array([x[:-1],x[1:]]).T.flatten()
    elif x.size == y.size :
        x1 = zeros(x.size+1)
        x1[1:-1] =          0.5*( x[1:]  + x[:-1] )
        x1[0]    = x1[1]  - 0.5*( x1[ 2] - x1[ 1] )
        x1[-1]   = x1[-2] + 0.5*( x1[-2] - x1[-3] )
        X = array([x1[:-1],x1[1:]]).T.flatten()
    else :
        raise ValueError( 'binplot: x.size must be equal to y.size or y.size+1' )

    Y = array([y,y]).T.flatten()

    if ax is None :
        if semilogy :
            plt.semilogy( X, Y, style, color=color, linewidth=linewidth, label=label )
        elif semilogx :
            plt.semilogx( X, Y, style, color=color, linewidth=linewidth, label=label )
        else :
            plt.plot( X, Y, style, color=color, linewidth=linewidth, label=label )
    else :
        if semilogy :
            ax.semilogy( X, Y, style, color=color, linewidth=linewidth, label=label )
        elif semilogx :
            ax.semilogx( X, Y, style, color=color, linewidth=linewidth, label=label )
        else :
            ax.plot( X, Y, style, color=color, linewidth=linewidth, label=label )
