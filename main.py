import unittest
import glob

import time

import itertools
import operator

# read one int per line from file
def read_ints(fname):
    return [int(line) for line in open(fname)]


# Day 1

def get_increases(data):
    return sum(map(lambda x: x[0] < x[1], itertools.pairwise(data)))

def sliding_window(data):
    return [sum(data[i: i+3]) for i in range(0, len(data) - 2)]

class Day1Test(unittest.TestCase):
    data = [199,
            200,
            208,
            210,
            200,
            207,
            240,
            269,
            260,
            263]

    def test_get_increases(self):
        self.assertEqual(7, get_increases(self.data))

    def test_sliding_window(self):
        self.assertEqual([607,618,618,617,647,716,769,792], sliding_window(self.data))

# sea floor sonar
def day1():
    data = read_ints('day1input.txt')

    start_time = time.time()

    task1 = get_increases(data)
    task2 = get_increases(sliding_window(data))

    return time.time() - start_time, task1, task2

# pilot the sub

def move_sub(pos, data):
    new_pos = pos.copy()
    for direction in data:
        x = direction[1]
        if direction[0] == 'forward':
            new_pos[0] += x
            new_pos[2] += x * new_pos[1]
        elif direction[0] == 'down':
            new_pos[1] += x
        elif direction[0] == 'up':
            new_pos[1] -= x

    return new_pos

class Day2Test(unittest.TestCase):
    data = [['forward', 5],
            ['down', 5],
            ['forward', 8],
            ['up', 3],
            ['down', 8],
            ['forward', 2]]

    def test_get_increases(self):
        pos = [0, 0, 0]
        self.assertEqual([15, 10, 60], move_sub(pos, self.data))
        self.assertEqual([0, 0, 0], pos)

def day2():
    data = [line.split() for line in open('day2input.txt')]
    data = map(lambda x: (x[0], int(x[1])), data)

    # cute - but it doubles the running time from 0.0003 s to 0.0007 s :-D
    # data = map(lambda x: (x[0], int(x[1])), map(operator.methodcaller("split"), open('day2input.txt')))

    start_time = time.time()

    pos = move_sub([0, 0, 0], data)
    task1 = pos[0] * pos[1]
    task2 = pos[0] * pos[2]

    return time.time() - start_time, task1, task2



def run(day):
    run_time, task1, task2 = day()
    print(day.__name__ + ": %.6s s - " % run_time + str(task1) + " " + str(task2))

def run_tests():
    test_files = glob.glob('*.py')
    module_strings = [test_file[0:len(test_file) - 3] for test_file in test_files]
    suites = [unittest.defaultTestLoader.loadTestsFromName(test_file) for test_file in module_strings]
    test_suite = unittest.TestSuite(suites)
    unittest.TextTestRunner().run(test_suite)


if __name__ == '__main__':
    run_tests()
    for i in range(1, 3):
        run(eval("day" + str(i)))