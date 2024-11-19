#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  K E N D A P Y . T I M E 1 4
#  class representing 14-digit date/time values
#
#  2016.10 L.Scheck 

from __future__ import absolute_import, division, print_function
import datetime

class Time14(object):
    """represents 14-digit date/time values"""

    time_int = 0

    def __init__( self, value, delta=None, format=None ) :
        """Initialize Time14 object from <value>
        units = auto[default] | sec | min | hour | 14 | 8
        delta = True/False : interpret value as time difference / absolute time"""

        dbg = False

        if format is None :
            format = 'auto'

        if isinstance(value, str) : # value is a string
            if dbg : print(('Time14:init: string ', value, ' length=', len(value)))
            l = len(value)
            if l == 14 :
                if dbg : print('guessing format = 14')
                self.time_int = int(value)
            elif l < 14 :
                if l >= 8 : # probably seconds (and maybe minutes and hours) are missing
                    if dbg : print('guessing format = 14')
                    if value[0] == '0' : # looks like a relative time
                        self.time_int = int('0'*(14-l) + value)
                    else : # is probably a absolute time
                        self.time_int = int(value + '0'*(14-l))
                else : # probably this is a time difference in seconds
                    t = int(value)
                    h = t // 3600
                    m = (t - h*3600) // 60
                    s = t - h*3600 - m*60
                    self.time_int = int( "%02d%02d%02d" % (h,m,s) )
            elif l > 14 :
                raise ValueError('Time14 string cannot be longer than 14 digits')

        elif isinstance(value,tuple) or isinstance(value,list) : # tuple/list with components
            if dbg : print(('Time14:init: tuple ', value))
            if len(value) >= 3 :
                Y = value[0]
                M = value[1]
                D = value[2]
            else :
                raise ValueError('Need at least 3 components to define a Time14')
            if len(value) >= 4 :
                h = value[3]
            else :
                h = 0
            if len(value) >= 5 :
                m = value[4]
            else :
                m = 0
            if len(value) == 6 :
                s = value[5]
            else :
                s = 0
            if len(value) > 6 :
                raise ValueError('More than 6 components specified for Time14')
            self.time_int = int( self.components2str(Y,M,D,h,m,s) )

        else : # value is a number
            if dbg : print(('Time14:init: num. val. ', value))

            if format == 'min' :
                s = 0
                m = value
                h = (value*60 - m*60 -s)//3600
                d = (value*60 - h*3600 - m*60 - s)//(24*3600)
                #print 'VALUE', value, d, h, m, s
                self.time_int = s + 100*m + 10000*h + 1000000*d
            else :
                if value > 9999999 :
                    self.time_int = int(value)

                else : # time in seconds
                    s = value % 60
                    m = ((value - s) // 60) % 60
                    h = (value - m*60 -s)//3600
                    d = (value - h*3600 - m*60 - s)//(24*3600)
                    #print 'VALUE', value, d, h, m, s
                    self.time_int = s + 100*m + 10000*h + 1000000*d
        if dbg :
            if self.is_delta() :
                print(('I am delta ', self.time_int))
            else :
                print(('I am absolute ', self.time_int))

    def __str__(self):
        if self.is_delta() :
            # convert to seconds
            #Y,M,D,h,m,s = self.components()
            #return "%08d" % ( s + 60*m + 3600*h )
            # 8 digit string
            return self.string(format='$D$h$m$s')
        else :
            # 14 digit string
            return "%014d" % self.time_int

    def components( self ) :
        s = "%014d" % self.time_int
        #      YYYY         MM           DD           hh            mm             ss
        return int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]), int(s[10:12]), int(s[12:14])

    def is_delta(self) :
        """Returns true for time differences (<-> year == 0)"""
        return self.time_int <=  99999999

    def is_absolute(self) :
        return self.time_int >   99999999

    def __add__(self, other):
        Y1,M1,D1,h1,m1,s1 = self.components()
        Y2,M2,D2,h2,m2,s2 = other.components()

        dtr = None
        if other.is_delta() :
            if self.is_delta() :
                dt1 = datetime.timedelta( D1, h1*3600 + m1*60 + s1 )
                dt2 = datetime.timedelta( D2, h2*3600 + m2*60 + s2 )
                dtr = dt1 + dt2
            else :
                t1 = datetime.datetime(Y1,M1,D1,h1,m1,s1)
                dt2 = datetime.timedelta( D2, h2*3600 + m2*60 + s2 )
                tr = t1 + dt2

        elif self.is_delta() :
            t2 = datetime.datetime(Y2,M2,D2,h2,m2,s2)
            dt1 = datetime.timedelta( D1, h1*3600 + m1*60 + s1 )
            tr = dt1 + t2

        else :
            raise ValueError("Time14: Cannot add two absolute times")
        # FIXME : adding two timedeltas should be possible

        if dtr is None :
            return Time14(tr.strftime("%Y%m%d%H%M%S"))
        else :
            return Time14(dtr.seconds)

    def __sub__(self, other):
        Y1,M1,D1,h1,m1,s1 = self.components()
        Y2,M2,D2,h2,m2,s2 = other.components()

        if self.is_delta() and other.is_delta() :
            dt1 = datetime.timedelta( D1, h1*3600 + m1*60 + s1 )
            dt2 = datetime.timedelta( D2, h2*3600 + m2*60 + s2 )
            tr = Time14( (D1-D2)*24*3600 + (h1-h2)*3600 + (m1-m2)*60 + (s1-s2) )

        elif (not self.is_delta()) and other.is_delta() :
            t1 = datetime.datetime(Y1,M1,D1,h1,m1,s1)
            dt2 = datetime.timedelta( D2, h2*3600 + m2*60 + s2 )
            tsub = t1 - dt2
            tr = Time14(tsub.strftime("%Y%m%d%H%M%S"))

        elif self.is_delta() and (not other.is_delta()) :
            raise ValueError("Time14: I do not want to subtract an absolute time from a time difference")

        else : # both absolute
            t1 = datetime.datetime(Y1,M1,D1,h1,m1,s1)
            t2 = datetime.datetime(Y2,M2,D2,h2,m2,s2)
            tsub = t1 - t2
            #print 'TSUB', str(tsub), int(tsub.total_seconds())
            tr = Time14( int(tsub.total_seconds()) )
        return tr

    def __lt__(self, other):
        return self.time_int < other.time_int

    def ___le__(self, other):
        return self.time_int <= other.time_int

    def __eq__(self, other):
        return self.time_int == other.time_int

    def __ne__(self, other):
        return self.time_int != other.time_int

    def __gt__(self, other):
        return self.time_int > other.time_int

    def __ge__(self, other):
        return self.time_int >= other.time_int   

    def year(self) :
        Y,M,D,h,m,s = self.components()
        return Y

    def month(self) :
        Y,M,D,h,m,s = self.components()
        return M

    def day(self) :
        Y,M,D,h,m,s = self.components()
        return D

    def hour(self) :
        Y,M,D,h,m,s = self.components()
        return h

    def minute(self) :
        Y,M,D,h,m,s = self.components()
        return m

    def second(self) :
        Y,M,D,h,m,s = self.components()
        return s

    def daysec(self) :
        Y,M,D,h,m,s = self.components()
        return s + 60*m + 3600*h

    def daymin(self) :
        Y,M,D,h,m,s = self.components()
        return m + 60*h

    def dayhour(self) :
        Y,M,D,h,m,s = self.components()
        return s/3600.0 + m/60.0 + h

    def string(self,format='$Y/$M/$D $h:$mUTC') :
        Y,M,D,h,m,s = self.components()
        f = format+''
        return f.replace('$Y','%04d'%Y).replace('$M','%02d'%M).replace('$D','%02d'%D \
               ).replace('$h','%02d'%h).replace('$m','%02d'%m).replace('$s','%02d'%s).replace('$y','%02d'%(Y-2000))

    def epoch_sec(self) :
        """return seconds since 1.1.1970 0UTC"""
        Y,M,D,h,m,s = self.components()
        t = datetime.datetime(Y,M,D,h,m,s)
        return (t-datetime.datetime.utcfromtimestamp(0)).total_seconds()

    def divisible(self, dt, ge=False) :
        """return last time earlier (or later, if ge=True) that day which can be divided by dt"""
        daysec = (self.daysec() // dt.daysec()) * dt.daysec()
        if ge :
            if daysec < self.daysec :
                daysec += dt.daysec()
        if daysec < 24*3600 :
            t = self
        else :
            daysec -= 24*3600
            t = self + Time14(24*3600)

        Y,M,D,h,m,s = t.components()
        h,m,s = t.sec2hms( daysec )
        return Time14(t.components2str(Y,M,D,h,m,s))


    def sec2hms(self, t) :
        h = t // 3600
        m = (t - h*3600) // 60
        s = t - h*3600 - m*60
        return h,m,s
        
    def components2str(self,Y,M,D,h,m,s) :
        return "%04d%02d%02d%02d%02d%02d" % (Y,M,D,h,m,s)

    def string14(self) :
        return self.string(format='$Y$M$D$h$m$s')

#--------------------------------------------------------------------------------
def time_range( t1, t2, deltat ) :
    """Returns list of times between t1 and t2 with spacing deltat"""

    if isinstance( t1, Time14 ) :
        t_i = t1
    else :
        t_i = Time14(t1)

    if isinstance( t2, Time14 ) :
        t_e = t2
    else :
        t_e = Time14(t2)

    if isinstance( deltat, Time14 ) :
        dt = deltat
    else :
        dt  = Time14(deltat)

    timelist = [str(t_i)]
    t = t_i
    while t < t_e :
        #print '>>> ', t
        t = t + dt
        timelist.append(str(t))

    return timelist

#--------------------------------------------------------------------------------
class Timeaxis(object) :
    """
    Converts absolut date/time values to plot coordinates and generates useful tick marks
    """
    def __init__( self ) :
        self.tmn_sec = 0     # minimum plot time
        self.tmx_sec = 0     # maximum plot time
        self.tmn_abs = None  # minimum absolute time
        self.tmx_abs = None  # maximum absolute time
        self.tref_sec = None # reference time in epoch seconds

    def convert( self, tin, format=None ) :
        """
        Convert absolute to plot times, remember maxima/minima
        """
        if type(tin) == list :
            tout = [ self.convert(t,format=format) for t in tin ]
        else :
            t14 = Time14(tin)
            if self.tref_sec is None :
                self.tref_sec = t14.epoch_sec()
                tout = 0
                self.tmn_abs = t14.string14()
                self.tmx_abs = t14.string14()
            else :
                tout = t14.epoch_sec() - self.tref_sec
                if tout < self.tmn_sec :
                    self.tmn_sec = tout
                    self.tmn_abs = t14.string14()
                if tout > self.tmx_sec :
                    self.tmx_sec = tout
                    self.tmx_abs = t14.string14()
            #print tin, tout, 'min/max sec ', self.tmn_sec, self.tmx_sec, 'min/max abs ', self.tmn_abs, self.tmx_abs, 'tref', self.tref_sec
        return tout

    def set_tickmarks(self,ax,margin=0.0):

        if self.tref_sec is None :
            return

        # determine tick interval
        duration = (self.tmx_sec-self.tmn_sec)/3600.0
        if duration <= 24 :
            dt_maj = 3
        elif duration <= 48 :
            dt_maj = 6
        elif duration <= 96 :
            dt_maj = 12
        else :
            dt_maj = 24

        dt = Time14(dt_maj*3600)
        #print 'dt = ', dt, dt.is_delta()

        tfirst = Time14(self.tmn_abs).divisible( dt, ge=False )
        tlast  = Time14(self.tmx_abs).divisible( dt, ge=False )
        t_abs = time_range( tfirst, tlast, dt )
        #print 't_abs = ', t_abs

        t_rel = []
        t_label = []
        for t in t_abs :
            #print '<<<', Time14(t)
            t_rel.append( Time14(t).epoch_sec() - self.tref_sec )
            t_label.append( '%d' % Time14(t).dayhour() )

        ax.set_xlim(( self.tmn_sec - margin*(self.tmx_sec-self.tmn_sec), self.tmx_sec + margin*(self.tmx_sec-self.tmn_sec) ))
        ax.set_xticks(t_rel)
        ax.set_xticklabels(t_label)

#-------------------------------------------------------------------------------
if __name__ == "__main__": # ---------------------------------------------------
#-------------------------------------------------------------------------------

    print()
    print('TIME14 TESTS')
    print()
    print('initialization from string:')
    for t in [ '20140516123517', '201405161235', '2014051612', '20140516', '10800','60'] :
        t14 = Time14(t)
        print("%14s = %s" % ( t, str(t14)))
        if t14.is_delta() :
            print('(time difference)')
        else :
            print('(absolute time)')
    print()
    print('addition')
    for ta, tb in [['201405161235','60'],['3600','2014051612']] :
        t14a = Time14(ta)
        t14b = Time14(tb)
        t14 = t14a + t14b
        print('%14s + %14s = %s' % (ta, tb, str(t14)), end=' ')
        if t14.is_delta() :
            print('(time difference)')
        else :
            print('(absolute time)')
    print()
    print('subtraction')
    for ta, tb in [['201405161235','60'],
                   ['201405161235','201405161135'],
                   ['201405161135','201405161235']] :
        t14a = Time14(ta)
        t14b = Time14(tb)
        t14 = t14a - t14b
        print('%14s - %14s = %s' % (ta, tb, str(t14)), end=' ')
        if t14.is_delta() :
            print('(time difference)')
        else :
            print('(absolute time)')
    print()
    print('time range')
    t1 = '201405162235'
    t2 = '201405170135'
    deltat = 1800
    print(('start time %s, end time %s, intervall %d sec' % (t1, t2, deltat)))
    tlist = time_range( t1, t2, deltat )
    print('--> ', end=' ')
    for t in tlist :
        print(t, ' ', end=' ')
    print()
