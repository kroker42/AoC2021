import unittest
import glob

import time

import itertools
from collections import namedtuple, Counter


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


# Day 3 diagnostics

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
        elif not common:
            keepers.append(i)

    return [data[p] for p in keepers]


def scrub(common, data):
    scrubbed_data = data.copy()

    for i in range(0, len(data[0])):
        scrubbed_data = scrub_list(common, i, scrubbed_data)
        if len(scrubbed_data) == 1:
            return scrubbed_data[0]


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
        epsilon = ''.join('1' if x == '0' else '0' for x in gamma)
        self.assertEqual(22, int(gamma, 2))
        self.assertEqual(9, int(epsilon, 2))

    def test_oxygen_scrub(self):
        self.assertEqual(['11110', '10110', '10111', '10101', '11100', '10000', '11001'],
                         scrub_list(True, 0, self.data))
        self.assertEqual('10111', scrub(True, self.data))

    def test_co2_scrub(self):
        self.assertEqual(['00100', '01111', '00111', '00010', '01010'],
                         scrub_list(False, 0, self.data))
        self.assertEqual('01010', scrub(False, self.data))


def day3():
    data = [line.strip() for line in open("day3input.txt")]

    start_time = time.time()

    gamma = ''.join([check_frequency(i, data) for i in range(0, len(data[0]))])
    epsilon = ''.join('1' if x == '0' else '0' for x in gamma)
    task1 = int(gamma, 2) * int(epsilon, 2)

    oxygen = scrub(True, data)
    co2 = scrub(False, data)
    task2 = int(oxygen, 2) * int(co2, 2)

    return time.time() - start_time, task1, task2


# Day 4 - Squid Bingo

def mark_board(num, board):
    for i in range(0, len(board)):
        board[i] = [-1 if x == num else x for x in board[i]]
        if sum(board[i]) == -5:
            return True

    for i in range(0, len(board[0])):
        if sum([x for x in [row[i] for row in board]]) == -5:
            return True

    return False


def score_board(board):
    return sum([x if x > 0 else 0 for x in itertools.chain.from_iterable(board)])


def bingo(drawn, boards):
    for d in drawn:
        for board in boards:
            if mark_board(d, board):
                return d, board


def last_winner(drawn, boards):
    winners = {i: 0 for i in range(0, len(boards))}

    for d in drawn:
        for i, won in winners.items():
            if not won and mark_board(d, boards[i]):
                winners[i] = d

        if list(winners.values()).count(0) == 0:
            for x in reversed(drawn):
                if x in winners.values():
                    return x, list(winners.keys())[list(winners.values()).index(x)]


class Day4Test(unittest.TestCase):
    data = [7, 4, 9, 5, 11, 17, 23, 2, 0, 14, 21, 24, 10, 16, 13, 6, 15, 25, 12, 22, 18, 20, 8, 19, 3, 26, 1]
    boards = [
        [[22, 13, 17, 11, 0],
         [8, 2, 23, 4, 24],
         [21, 9, 14, 16, 7],
         [6, 10, 3, 18, 5],
         [1, 12, 20, 15, 19]],

        [[3, 15, 0, 2, 22],
         [9, 18, 13, 17, 5],
         [19, 8, 7, 25, 23],
         [20, 11, 10, 24, 4],
         [14, 21, 16, 12, 6]],

        [[14, 21, 17, 24, 4],
         [10, 16, 15, 9, 19],
         [18, 8, 23, 26, 20],
         [22, 11, 13, 6, 5],
         [2, 0, 12, 3, 7]]
    ]

    def test_mark_board_row(self):
        board = self.boards[0].copy()
        self.assertFalse(mark_board(22, board))
        self.assertEqual(-1, board[0][0])
        mark_board(13, board)
        mark_board(17, board)
        mark_board(11, board)
        self.assertTrue(mark_board(0, board))
        self.assertEqual([-1] * 5, board[0])

    def test_mark_board_col(self):
        board = self.boards[0].copy()
        mark_board(13, board)
        mark_board(2, board)
        mark_board(9, board)
        mark_board(10, board)
        self.assertTrue(mark_board(12, board))
        self.assertEqual([-1] * 5, [row[1] for row in board])

    def test_bingo(self):
        num, winner = bingo(self.data, [b.copy() for b in self.boards])
        self.assertEqual(24, num)
        self.assertEqual([[-1] * 5,
                          [10, 16, 15, -1, 19],
                          [18, 8, -1, 26, 20],
                          [22, -1, 13, 6, -1],
                          [-1, -1, 12, 3, -1]], winner)
        self.assertEqual(188, score_board(winner))

    def test_last_winner(self):
        d, winner = last_winner(self.data, [b.copy() for b in self.boards])
        self.assertEqual(13, d)
        self.assertEqual(1, winner)


def day4():
    data = [line.split() for line in open('day4input.txt')]
    drawn = [int(d) for d in data[0][0].split(',')]
    boards = []

    for line in data[1:]:
        if not line:
            boards.append([])
        else:
            boards[-1].append([int(d) for d in line])

    start_time = time.time()

    num, winner = bingo(drawn, boards)
    task1 = num * score_board(winner)

    last_num, last = last_winner(drawn[drawn.index(num) + 1:], boards)
    task2 = last_num * score_board(boards[last])

    return time.time() - start_time, task1, task2


# Day 5 - Vent labyrinth

pt = namedtuple('pt', 'x y')
vent = namedtuple('vent', 'pt1 pt2')

def vents(str):
    x, y = [p.strip().split(',') for p in str.split('->')]
    return vent(pt(int(x[0]), int(x[1])), pt(int(y[0]), int(y[1])))

def add_vent_line(m, v):
    x_step = -1 if v.pt1.x > v.pt2.x else 1 if v.pt1.x < v.pt2.x else 0
    y_step = -1 if v.pt1.y > v.pt2.y else 1 if v.pt1.y < v.pt2.y else 0

    no_iter = max(abs(v.pt1.x - v.pt2.x), abs(v.pt1.y - v.pt2.y))

    for i in range(no_iter + 1):
        p = pt(v.pt1.x + i * x_step, v.pt1.y + i * y_step)
        m[p] = m[p] + 1 if p in m else 1


def vent_map(grid):
    m = {}
    for v in grid:
        if v.pt1.x == v.pt2.x or v.pt1.y == v.pt2.y:
            add_vent_line(m, v)
    return m


def vent_map_3D(grid, m):
    for v in grid:
        if v.pt1.x != v.pt2.x and v.pt1.y != v.pt2.y:
            add_vent_line(m, v)
    return m


class Day5Test(unittest.TestCase):
    input = ['0,9 -> 5,9',
            '8,0 -> 0,8',
            '9,4 -> 3,4',
            '2,2 -> 2,1',
            '7,0 -> 7,4',
            '6,4 -> 2,0',
            '0,9 -> 2,9',
            '3,4 -> 1,4',
            '0,0 -> 8,8',
            '5,5 -> 8,2']

    grid = [vents(x) for x in input]

    def test_vents(self):
        self.assertEqual(vent(pt(0, 9), pt(5, 9)), vents(self.input[0]))

    def test_vent_map(self):
        v = vent_map(self.grid)
        self.assertEqual(1, v[pt(7, 0)])
        self.assertEqual(2, v[pt(7, 4)])
        self.assertEqual(5, list(v.values()).count(2))

        vent_map_3D(self.grid, v)
        self.assertEqual(1, v[pt(0, 0)])
        self.assertEqual(12, len(v) - list(v.values()).count(1))

    def test_add_vent_line(self):
        m = {}
        add_vent_line(m, self.grid[0])
        self.assertEqual(1, m[pt(0, 9)])
        self.assertEqual(1, m[pt(5, 9)])

        add_vent_line(m, self.grid[1])
        self.assertEqual(1, m[pt(8, 0)])
        self.assertEqual(1, m[pt(6, 2)])
        self.assertEqual(1, m[pt(0, 8)])



def day5():
    data = [vents(line) for line in open('day5input.txt')]
    start_time = time.time()

    v = vent_map(data)
    task1 = len(v) - list(v.values()).count(1)

    vent_map_3D(data, v)
    task2 = len(v) - list(v.values()).count(1)

    return time.time() - start_time, task1, task2


# Day 6 - exponential fish populations

def spawn(fish):
    new_fish = []
    for f in fish:
        new_f = (f-1) % 7 if f < 7 else f-1
        new_fish.append(new_f)
        if f == 0:
            new_fish.append(8)
    return new_fish

def iter_fish(fish):
    new_fish = {i: 0 for i in range(9)}
    for f, n in fish.items():
        new_f = (f - 1) % 7 if f < 7 else f - 1
        new_fish[new_f] += n
        if f == 0:
            new_fish[8] += n

    return new_fish

def smarter_spawn(fish, days):
    new_fish = fish
    for i in range(days):
        new_fish = iter_fish(new_fish)

    return new_fish

class Day6Test(unittest.TestCase):
    data = [3, 4, 3, 1, 2]

    def test_spawn(self):
        fish = self.data
        for i in range(18):
            fish = spawn(fish)
        self.assertEqual(26, len(fish))

        for i in range(18, 80):
            fish = spawn(fish)
        self.assertEqual(5934, len(fish))

    def test_smarter_spawn(self):
        fish = dict(Counter(self.data))
        self.assertEqual(26, sum(smarter_spawn(fish, 18).values()))
        self.assertEqual(5934, sum(smarter_spawn(fish, 80).values()))


def day6():
    file = open('day6input.txt')
    data = [int(x) for x in file.readline().split(',')]

    start_time = time.time()

    # smarter_spawn runs in 0.0007s instead of 0.45s for spawn()
    fish = smarter_spawn(dict(Counter(data)), 80)
    task1 = sum(fish.values())

    fish = smarter_spawn(dict(Counter(data)), 256)
    task2 = sum(fish.values())

    return time.time() - start_time, task1, task2

# Day 7 - Crab Data Centre

def get_distances(crabs):
    distances = {}
    for i in range(len(crabs)):
        if crabs[i] not in distances:
            distances[crabs[i]] = 0
            for c in crabs:
                distances[crabs[i]] += abs(c - crabs[i])

    return distances

def get_weighted_distances(crabs):
    min_crab = min(crabs)
    max_crab = max(crabs)
    distances = {i : 0 for i in range(min_crab, max_crab + 1)}

    for c in crabs:
        for i in distances.keys():
            dist = abs(c - i)
            distances[i] += dist * (dist + 1) // 2

    return distances


class Day7Test(unittest.TestCase):

    data = [16, 1, 2, 0, 4, 2, 7, 1, 2, 14]
    def test_get_distances(self):
        distances = get_distances(self.data)
        self.assertEqual(37, distances[2])
        self.assertEqual(37, min(distances))

    def test_get_weighted_distances(self):
        distances = get_weighted_distances(self.data)
        self.assertEqual(206, distances[2])
        self.assertEqual(168, min(distances.values()))


def day7():
    file = open('day7input.txt')
    data = [int(x) for x in file.readline().split(',')]

    start_time = time.time()

    task1 = min(get_distances(data).values())
    task2 = min(get_weighted_distances(data).values())

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
    for i in range(1, 8):
        run(eval("day" + str(i)))
