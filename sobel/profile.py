#!/usr/bin/python
#-*- coding:utf-8 -*-

# Each executable that wants to output performance information must
# output it to a file containing an array of JSON objects of the form:
# { "name": "a description of the test",
#   "size": for instance number of elements, or size of the matrix
#   "nProcs": number of processors
#   "time": time in seconds in decimal form
# }
#
# This script must be called with fisrt argument being the key for X-axix
# (either 'size' or 'nprocs') and some files containing properly formatted
# JSON

import json
import matplotlib.pyplot as plt
import sys


class Test:
    def __init__(self, key, measures, color, name):
        self.key = key
        self.measures = measures
        self.color = color
        self.name = name



colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', '0.75', '#FF5100', '#780000',
          '#005078']

# get file names and criteria for X-axix
if len(sys.argv) < 3:
    print("Usage: %s [size|nProcs] file [file...]" % sys.argv[0])
    exit(1)

key = sys.argv[1]
filenames = sys.argv[2:]


# Get all the data in one big dict
data = []
for fname in filenames:
    d = json.load(open(fname))
    data += d


# Partition in different tests based on their name
tests = []
test_names = list(set( [ e['name'] for e in data ] ))
for i, name in enumerate(test_names):
    tests_for_name = [e for e in data if e['name'] == name]
    tests.append(Test(key, tests_for_name, colors[i], name))


# Now we can plot every test
for t in tests:
    n_items_or_procs = [ e[t.key] for e in t.measures ]
    times = [ e['time'] for e in t.measures ]

    plt.plot(n_items_or_procs, times,
             color = t.color, linestyle='-', marker='o',
             linewidth=2, label=t.name)


if key == "size":
    plt.xlabel("Number of pixels")
else:
    plt.xlabel("Number of execution cores")

plt.ylabel("Execution time in seconds")
plt.legend(loc='best', prop={'size':12})
plt.show()



#correspondance_color_files = ""

#for i, fname in enumerate(filenames):
    #d = json.load(open(fname))
    #n_iter = [v['nElems'] for v in d]
    #times = [v['time'] for v in d]

    #plt.plot(n_iter, times, color=colors[i], linestyle='-', marker='o', linewidth=2, label=fname)



#plt.xlabel("Number of elements in the sum")
#plt.ylabel("Execution time in seconds")
#plt.title("Quicksort\n" + correspondance_color_files)
#plt.legend(loc='best', prop={'size':12})

#plt.show()
