#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . B A C Y _ U T I L S
#
#  2020.6 L.Scheck 

from __future__ import absolute_import, division, print_function
import os, sys, subprocess, glob, tempfile, re
from datetime import datetime, timedelta, date, time
from math import fabs

# DATE/TIME MANAGEMENT #########################################################

#-------------------------------------------------------------------------------
def t2str( t, digits=14 ) :
    """Convert datetime object to 8, 10, 12 or 14-digit string"""
    if digits == 14 :
        s = t.strftime( '%Y%m%d%H%M%S' )
    elif digits == 12 :
        s = t.strftime( '%Y%m%d%H%M' )
    elif digits == 10 :
        s = t.strftime( '%Y%m%d%H' )
    elif digits == 8 :
        s = t.strftime( '%Y%m%d' )
    else :
        raise ValueError('Cannot generate datetime string')
    return s


#-------------------------------------------------------------------------------
def str2t( s, hour=None, minute=None, second=None ) :
    """Convert 8, 10, 12 or 14-digit string to datetime object"""
    if len(s) == 14 :
        s_ = s
    elif len(s) == 12 : # seconds missing
        s_ = s + '00'
    elif len(s) == 10 : # minutes missing
        s_ = s + '0000'
    elif len(s) == 8  : # hours missing
        s_ = s + '000000'
    else :
        raise ValueError('Cannot parse datetime string '+s)

    s_ = '{}{}{}{}'.format( s_[:8],
                            s_[8:10]  if hour   is None else '{:02d}'.format(hour),
                            s_[10:12] if minute is None else '{:02d}'.format(minute),
                            s_[12:14] if second is None else '{:02d}'.format(second) )

    return datetime.strptime( s_, '%Y%m%d%H%M%S' )

#------------------------------------------------------------------------------
def to_datetime( t ) :
    """Convert string, integer or datetime object to datetime object"""
    if type(t) == datetime :
        r = t
    else :
        r = str2t('{}'.format(t))
    return r

#------------------------------------------------------------------------------
def to_timedelta( t, units=None ) :
    """Convert string, integer or timedelta object to timedelta object"""

    if type(t) == timedelta :
        r = t
    else :
        t_ = None
        if units is None and isinstance(t,str) :
            for u in ['days','hours','minutes','seconds','min','sec','d','h','m','s'] :
                if t.endswith(u) :
                    units = u
                    t_ = t[:-len(u)].strip()
                    break
        if t_ is None :
            t_ = t

        if isinstance(t_,str) and ':' in t_ :
            if 'd' in t_ : # e.g. 1d12:00
                tknsd = t_.split('d')
                d = int(tknsd[0])
                tkns  = tknsd[1].split(':')
            else :
                d = 0
                tkns  = t_.split(':')
            if len(tkns) == 2 :
                r = timedelta( days=d, hours=float(tkns[0]), minutes=float(tkns[1]) )
            elif len(tkns) == 3 :
                r = timedelta( days=d, hours=float(tkns[0]), minutes=float(tkns[1]), seconds=float(tkns[2]) )
            else :
                raise ValueError('I cannot understand time delta '+t_)
        elif units is None or units in ['s','sec','seconds'] :
            r = timedelta( seconds=float('{}'.format(t_)) )
        elif units in ['min','minutes'] :
            r = timedelta( minutes=float('{}'.format(t_)) )
        elif units in ['h','hours'] :
            r = timedelta( hours=float('{}'.format(t_)) )
        elif units in ['d','days'] :
            r = timedelta( days=float('{}'.format(t_)) )
        else :
            raise ValueError('I do not understand the timedelta units')
    return r

#------------------------------------------------------------------------------
def midnight( d ) :
    """Return midnight datetime for given datetime"""
    
    d = d.date()
    t = time(0,0)
    return datetime.combine(d, t)

#------------------------------------------------------------------------------
def td2str( td, format='HH:MM' ) :
    """Convert timedelta object to string"""
    
    if format == 'HH:MM' :
        h = td.days*24 + td.seconds//3600
        m = (td.seconds % 3600)//60
        r = '{:02d}:{:02d}'.format(h,m)
    elif format == 'DDHHMMSS' :
        d = td.days
        h = td.seconds  // 3600
        m = (td.seconds %  3600) // 60
        s = td.seconds           %  60
        r = '{:02d}{:02d}{:02d}{:02d}'.format(d,h,m,s)
    else :
        raise ValueError('Unknown timedelta string format')
    return r


# DATA STRUCTURES ##############################################################

#-------------------------------------------------------------------------------
def add_branch( ft, ft_path, val, overwrite=False ) :
    """Add a branch and the corresponding value to a dictionary tree

        example: a = {}; add_branch( a, ['x','y','z'], 1 )
                 --> a == { 'x':{ 'y':{ 'z':1 } } }
    """
    ft_ = ft # start at root level
    
    if len(ft_path) > 1 : # there are intermediate path elements
        for elem in ft_path[:-1] : # process all but the last path element
            if not elem in ft_ :   # create missing part of the branch, if necessary
                ft_[elem] = {}
            ft_ = ft_[elem]        # descend one level
    
    if not overwrite and ft_path[-1] in ft_ :
        raise ValueError('Overwriting'+('-'.join(ft_path)))

    ft_[ft_path[-1]] = val         # set value for last path element

#-------------------------------------------------------------------------------
def common_subset( lists ) :
    """Return common subset of list elements as sorted list"""
    
    #cms = set()
    #for l in lists :
    #    cms.update(set(l))
    #return sorted(list(cms))

    cms = set(lists[0])
    for i in range(1,len(lists)) :
        cms.intersection_update( lists[i] )
    return sorted(list(cms))


# PLOTTING #####################################################################

def adjust_time_axis( fig, ax, dates, density=1 ) :
    """Adjust tick marks and labels for a plot generated with plot_date."""

    from matplotlib.dates import DayLocator, HourLocator, MinuteLocator, DateFormatter, drange

    ts = dates[0]  # start date
    te = dates[-1] # end   date
    td = (te - ts).total_seconds() / density

    # FIXME: Define more cases...

    if td < 24*3600 : # <= 1 day
        ax.xaxis.set_major_locator(HourLocator(range(0,25)))
        ax.xaxis.set_minor_locator(MinuteLocator(range(0, 60, 15)))
        ax.xaxis.set_major_formatter(DateFormatter('%b%d, %H:%MUTC'))
    elif td > 24*3600 and td <= 2*24*3600 : # 1-2 days
        ax.xaxis.set_major_locator(HourLocator(range(0, 25, 3)))
        ax.xaxis.set_minor_locator(HourLocator(range(0, 25, 1)))
        ax.xaxis.set_major_formatter(DateFormatter('%b%d, %HUTC'))
    elif td > 2*24*3600 and td <= 7*24*3600 : # 2 days - 1 week
        ax.xaxis.set_major_locator(HourLocator(range(0, 25, 6)))
        ax.xaxis.set_minor_locator(HourLocator(range(0, 25, 3)))
        ax.xaxis.set_major_formatter(DateFormatter('%b%d, %HUTC'))
    elif td > 7*24*3600 and td <= 2*7*24*3600 : # 1 - 2 weeks
        ax.xaxis.set_major_locator(HourLocator([0,12]))
        ax.xaxis.set_minor_locator(HourLocator(range(0, 25, 6)))
        ax.xaxis.set_major_formatter(DateFormatter('%b%d, %HUTC'))
    elif td > 2*7*24*3600 and td < 4*7*24*3600 : # 2 - 4 weeks
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_minor_locator(HourLocator(range(0, 25, 12)))
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    else : #  > 4 weeks
        ax.xaxis.set_major_locator(DayLocator(bymonthday=[1,8,15,21,30]))
        ax.xaxis.set_minor_locator(DayLocator(bymonthday=range(1,32)))
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    ax.fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')
    fig.autofmt_xdate()


#-------------------------------------------------------------------------------
def default_color_sequence() :
    import matplotlib.colors
    # see e.g. https://en.wikipedia.org/wiki/Web_colors#X11_color_names
    return [ matplotlib.colors.CSS4_COLORS[c] for c in [ 'royalblue', 'darkorange', 
             'darkviolet', 'darkcyan', 'green', 'brown', 'navy', 'deeppink',
             'orchid', 'darkgreen', 'teal', 'tomato', 'olive', 'gold' ] ]

#-------------------------------------------------------------------------------
def expand_color_name( s ) :
    import matplotlib.colors

    if len(s) == 1 : # r, g, b,...
        return s

    elif s.startswith('#') : #  e.g. #00ff00
        return s

    else : # named color, see https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        if s in matplotlib.colors.CSS4_COLORS :
            return matplotlib.colors.CSS4_COLORS[s]

        else :
            raise ValueError('Unknown color.')

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    col='royalblue'
    print( '{} = {}'.format( col, expand_color_name(col) ) )

