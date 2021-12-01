import unittest
import glob

import time


# read one int per line from file
def read_ints(fname):
    return [int(line) for line in open(fname)]


def get_increases(data):
    prev = data[0]
    cnt = 0
    for d in data[1:]:
        if d > prev:
            cnt += 1
        prev = d
    return cnt


class Day1Test(unittest.TestCase):

    def test_get_increases(self):
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
        pairs = get_increases(data)
        self.assertEqual(7, pairs)


# sea floor sonar -
def day1():
    data = read_ints('day1input.txt')

    start_time = time.time()

    task1 = get_increases(data)
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
    for i in range(1, 2):
        run(eval("day" + str(i)))
