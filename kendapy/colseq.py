#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . C O L S E Q
#  define sequence of clearly-distinguishable, equally bright (TODO) colors
#
#  2017.7 L.Scheck

from __future__ import absolute_import, division, print_function
import colorsys
from numpy import exp, sin, pi, array, arange

def colseq(i,n,dhue=0.0,retall=False,sat=0.8,no_green=False) :
    """Return color #i from a sequence of n colors."""

    # compute and normalize hue
    hue = (i+0.5)/n
    hue -= dhue + 0.28
    if n == 3 : hue -= 0.25
    if hue <  0.0 : hue += 1
    if hue >= 1.0 : hue -= 1

    # transform hue such that neighbouring colors are equally good distinguishable
    hue +=  0.1*exp(-((hue-0.3)/0.15)**2)*sin(0.5*pi*(hue-0.3)/0.15)
    hue +=  0.05*exp(-((hue-0.9)/0.05)**2)*sin(0.5*pi*(hue-0.9)/0.05)

    # avoid green hues
    if no_green :
        hlow = 0.18
        hhi  = 0.5
        hmid = 0.5*(hlow+hhi)
        if hue < hmid :
            hue *= hlow/hmid
        else :
            hue = hhi + (1.0-hhi)*(hue-hmid)/(1.0-hmid)

    # set luminosity as a function of hue to achieve similarly bright colors
    lum = 0.65
    lum -= 0.15*exp(-((hue-0.18)/0.1)**2)
    lum -= 0.10*exp(-((hue-0.35)/0.1)**2)
    lum -= 0.18*exp(-((hue-0.50)/0.1)**2)
    lum -= 0.10*exp(-((hue-0.80)/0.1)**2)

    # if there are many colors, use additional luminosity variation
    # to make neighbouring colors look more different
    if n > 7 :
        lum += 0.03 - 0.06*(i % 2)

    col = colorsys.hls_to_rgb( hue, lum, sat )

    if retall :
        return col, hue, lum, sat
    else :
        return col

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    # show sequences
    import pylab as plt
    plt.figure(1)
    plt.clf()
    nmax=16
    for n in range(nmax) :
        for i in range(n) :
            plt.plot( ( float(i)/n, float(i+1)/n ), (n/float(nmax),n/float(nmax)), color=colseq(i,n), linewidth=10 )
    n=30
    hue = array([ colseq(i,n,retall=True)[1] for i in range(n) ])
    lum = array([ colseq(i,n,retall=True)[2] for i in range(n) ])
    plt.plot( arange(n)/float(n-1), hue-1, color='r' )
    plt.plot( arange(n)/float(n-1), lum-2, color='k' )
    plt.grid()
    plt.savefig('colseq.png')
