import unittest
import glob

import time

import itertools

# read one int per line from file
def read_ints(fname):
    return [int(line) for line in open(fname)]


# Day 1 - create map of sea bed

def get_increases(data):
    return sum(map(lambda x: x[0] < x[1], itertools.pairwise(data)))


def sliding_window(data):
    return [sum(data[i: i + 3]) for i in range(0, len(data) - 2)]


class Day1Test(unittest.TestCase):
    data = [199, 200, 208, 210, 200, 207, 240, 269, 260, 263]

    def test_get_increases(self):
        self.assertEqual(7, get_increases(self.data))

    def test_sliding_window(self):
        self.assertEqual([607, 618, 618, 617, 647, 716, 769, 792], sliding_window(self.data))

def day1():
    data = read_ints('day1input.txt')

    start_time = time.time()

    task1 = get_increases(data)
    task2 = get_increases(sliding_window(data))

    return time.time() - start_time, task1, task2


# Day 2 - pilot the sub

def move_sub(pos, data):
    new_pos = list(pos)
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
    data = [['forward', 5], ['down', 5], ['forward', 8], ['up', 3], ['down', 8], ['forward', 2]]

    def test_get_increases(self):
        pos = (0, 0, 0)
        self.assertEqual([15, 10, 60], move_sub(pos, self.data))


def day2():
    data = [line.split() for line in open('day2input.txt')]
    data = map(lambda x: (x[0], int(x[1])), data)

    start_time = time.time()

    pos = move_sub((0, 0, 0), data)
    task1 = pos[0] * pos[1]
    task2 = pos[0] * pos[2]

    return time.time() - start_time, task1, task2

# Day 3

def check_frequency(pos, data):
    frequency = 0
    for d in data:
        frequency += int(d[pos])

    return '1' if frequency >= len(data) / 2 else '0'


def scrub_list(common, pos, data):
    keepers = []
    most_common = check_frequency(pos, data)
    for i in range(0, len(data)):
        if data[i][pos] == most_common:
            if common:
                keepers.append(i)
    return [data[p] for p in keepers]

def scrub(common, data):
    scrubbed_data = data.copy()

    for i in range(0, len(data[0])):
        scrubbed_data = scrub_list(common, i, scrubbed_data)
        if len(scrubbed_data) == 1:
            return scrubbed_data


class Day3Test(unittest.TestCase):
    data = ['00100',
            '11110',
            '10110',
            '10111',
            '10101',
            '01111',
            '00111',
            '11100',
            '10000',
            '11001',
            '00010',
            '01010']
    def test_check_frequency(self):
        self.assertEqual('1', check_frequency(0, self.data))
        self.assertEqual('0', check_frequency(1, self.data))

    def test_gamma(self):
        gamma = ''.join([check_frequency(i, self.data) for i in range(0, 5)])
        self.assertEqual(22, int(gamma, 2))
        epsilon = ''.join('1' if x == '0' else '0' for x in gamma)
        self.assertEqual(9, int(epsilon, 2))

    def test_scrub(self):
        self.assertEqual(['11110', '10110', '10111', '10101', '11100', '10000', '11001'],
                         scrub_list(True, 0, self.data))
        self.assertEqual(['10111'], scrub(True, self.data))


def day3():
    data = [line.strip() for line in open("day3input.txt")]

    start_time = time.time()

    gamma = ''.join([check_frequency(i, data) for i in range(0, len(data[0]))])
    epsilon = ''.join('1' if x == '0' else '0' for x in gamma)

    task1 = int(gamma, 2) * int(epsilon, 2)
    task2 = None

    return time.time() - start_time, task1, task2

# Day

class DayTest(unittest.TestCase):


    def test_(self):
        self.assertEqual(True, True)


def day():
    data = [line.split() for line in open('dayinput.txt')]
    start_time = time.time()

    task1 = None
    task2 = None

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
    for i in range(1, 4):
        run(eval("day" + str(i)))



