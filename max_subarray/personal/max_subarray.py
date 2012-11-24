#!/usr/bin/python
#-*- coding: utf-8 -*-


import sys
import re
from copy import deepcopy
from termcolor import cprint, colored


def cumul_lines_in_columns(m):
    """Given a matrix, returns the matrix which lines are the cumulative
       sum of the initial matrix. For instance:
           1 -5 0               1 -5 0
           1  3 6       ==>     2 -2 6
           2  9 4               4  7 10 """
    width = len(m[0])
    height = len(m)
    cumul = deepcopy(m)
    for i in range(1, height):
        for j in range(0, width):
            cumul[i][j] += cumul[i-1][j]

    return cumul

class matrix_area:
    def __init__(self, x_min, y_min, x_max, y_max):
        assert(x_min <= x_max and y_min <= y_max)
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max



def max_subarray(m):
    width = len(m[0])
    height = len(m)

    cumul = cumul_lines_in_columns(m)
    cur_max = m[0][0], matrix_area(0, 0, 0, 0)

    # On cherche le subarray le plus grand pour chaque ensemble de lignes
    # [i:k]. On utilise notre cumul pour dire que le plus grand subarray
    # entre les lignes i et k a pour valeur et pour limites :
    #               kandane(cumul[k][:] - cumul[i-1][:])
    # car cumul[k] - cumul[i-1] est la ligne contenant le cumul des lignes
    # entre k et i (0..k - 0..i-1)
    for i in range(height):
        for k in range(i, height):
            if i == 0:
                between_i_k = cumul[k]
            else:
                between_i_k = [a - b for (a, b) in zip(cumul[k], cumul[i-1])]

            loc_max, loc_start, loc_end = kandane(between_i_k)

            # Did we get better ?
            if loc_max > cur_max[0]:
                #print("Update: min: (%d, %d), max: (%d, %d)" % (loc_start, i, loc_end, k))
                cur_max = loc_max, matrix_area(loc_start, i, loc_end, k)


    return cur_max





def kandane(array):
    max_sum, start, end = min(array) - 1, 0, 0
    cur_max = 0
    cur_start = 0

    for cur_end in range(0, len(array)):
        cur_max += array[cur_end]
        if cur_max > max_sum:
            #print("update: cur_max: %d, max_sum: %d" % (cur_max, max_sum))
            #print("cur_start: %d, cur_end: %d" % (cur_start, cur_end))
            max_sum, start, end = cur_max, cur_start, cur_end

        if cur_max < 0:
            #print("We are negative, set cur_start = %d" % (cur_end + 1))
            cur_max = 0
            cur_start = cur_end + 1

    return max_sum, start, end


def pretty_print(m, a):
    width = len(m[0])
    height = len(m)

    def digit_length(n):
        return len(str(n))

    def flatten(l):
            return list(item for iter_ in l for item in iter_)

    longest_digit_length = max (map(digit_length, flatten(m)))

    colored_str = ""

    for i in range(height):
        for j in range(width):
            pad = longest_digit_length - digit_length(m[i][j]) + 1
            if i >= a.y_min and i <= a.y_max and j >= a.x_min and j <= a.x_max:
                colored_str += colored(m[i][j], 'blue', 'on_grey') + pad * " "
            else:
                colored_str += colored(m[i][j], 'white', 'on_grey') + pad * " "
        colored_str += "\n"

    cprint(colored_str)



def parse_file(filename):
    f = open(filename)
    lines = f.readlines()
    
    # First line gives the number of lines and columns
    size = int(lines[0])

    mat = [[] for i in range(size)]
    
    assert (len(lines) == size + 1)

    
    for i,l in enumerate(lines[1:]):
        #print(re.split(r"\s+", l[:-1]))
        splits = re.split(r"\s+", l.strip())
        nums = [int(x) for x in splits]
        assert (len(nums) == size)
        mat[i] = nums

    return mat


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: %s filename" % sys.argv[0])

    m = parse_file(sys.argv[1])
    maximum, area = max_subarray(m)

    #pretty_print(m, area)
    print("P: Max value: %d, between (%d, %d) and (%d, %d)" % (maximum, area.x_min, area.y_min, area.x_max, area.y_max))
