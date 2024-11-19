#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from matplotlib import pyplot as plt
from numpy import *
import sys
from kendapy.ekf import Ekf, tables

# Beispiel: FOF-Datei mit ekf.py lesen, Beobachtungen filtern

fof = Ekf(sys.argv[1],verbose=True)         # Ekf Objekt erzeugen

plt.figure(1,figsize=(10,10))

fof.set_filter(state_filter='all')    # keine Einschränkungen bzgl. Status der Beobachtung
plt.scatter( fof.obs(param='lon'), fof.obs(param='lat'), c='#999999', s=20, linewidths=0 )

fof.set_filter(state_filter='passive') # nur passive Beobachtungen
plt.scatter( fof.obs(param='lon'), fof.obs(param='lat'), c='#990000', s=10, linewidths=0 )

fof.set_filter(state_filter='active')  # nur aktive Beobachtungen (das ist der default, wenn man nicht explizit einen Filter definiert)
plt.scatter( fof.obs(param='lon'), fof.obs(param='lat'), c='r', s=10, linewidths=0 )

fof.set_filter(state_filter='active',area_filter='LATLON:47.7,3.5,56.0,17.5')  # nur aktive Beobachtungen in einem bestimmten Gebiet
plt.scatter( fof.obs(param='lon'), fof.obs(param='lat'), c='b', s=10, linewidths=0 )

# globalen Filter zurücksetzen (-> Gebietsbeschränkung ist wieder aufgehoben)
fof.set_filter(state_filter='active')


# Man kann auch Filtereinstellungen direkt angeben (sie wirken dann zusätzlich zum global aktiven Filter, also jetzt gerade nur state='active')

fkw = { 'obstype_filter':'AIREP', 'varname_filter':'U' } # nur AIREP U-Geschwindigkeiten zulassen
plt.scatter( fof.obs(param='lon',**fkw), fof.obs(param='lat',**fkw), c='g', s=10, linewidths=0 )

plt.savefig('state_map.png')

# so kommt man an Zeit und Höhe:
plt.clf()
fkw = { 'obstype_filter':'AIREP' }
plt.scatter( fof.obs(param='time',**fkw), fof.obs(param='plevel',**fkw)/100.0, s=10, c='b', linewidths=0, label='AIREP' )
fkw = { 'obstype_filter':'TEMP', 'state_filter':'all' }
plt.scatter( fof.obs(param='time',**fkw), fof.obs(param='plevel',**fkw)/100.0, s=50, c='r', linewidths=0, label='TEMP' )
plt.legend()
plt.ylim((1000,0))
plt.xlabel('t - t_ref [min]')
plt.ylabel('P [hPa]')
plt.savefig('time_plevel.png')

# Und an die beobachteten Werte und die Modelläquivalente kommt man so:
plt.clf()
fkw = { 'obstype_filter':'AIREP', 'varname_filter':'T', 'state_filter':'all' }
plt.scatter( fof.obs(**fkw) - fof.fg(**fkw), fof.obs(param='plevel',**fkw)/100.0, s=20, c='#999999', linewidths=0 )
fkw = { 'obstype_filter':'AIREP', 'varname_filter':'T', 'state_filter':'active' }
plt.scatter( fof.obs(**fkw) - fof.fg(**fkw), fof.obs(param='plevel',**fkw)/100.0, s=10, c='r', linewidths=0 )
plt.legend()
plt.ylim((1000,0))
plt.xlabel('T_obs - T_fg [K]')
plt.ylabel('P [hPa]')
plt.savefig('obs_fg.png')

