import unittest
import glob

import time

import operator
import itertools
import statistics
from collections import namedtuple, Counter, deque
from functools import reduce

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
    distances = {i: 0 for i in range(min_crab, max_crab + 1)}

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
        self.assertEqual(37, min(distances.values()))

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


# Day 8 - number display

def get_display_numbers(display):
    data = []
    for line in display:
        data.append([x.strip().split(' ') for x in line.split('|')])
    return data


def count_output_digits(data):
    cnt = 0
    for d in data:
        cnt += sum(map(lambda x: len(x) in [2, 3, 4, 7], d[1]))
    return cnt


def get_segment_map(data):
    digits = {}

    for d in data:
        if len(d) in digits.keys():
            digits[len(d)].append(d)
        else:
            digits[len(d)] = [d]

    segment_map = {}
    segment_map['a'] = set(digits[3][0]) - set(digits[2][0])
    segment_map['cf'] = set(digits[3][0]).intersection(set(digits[2][0]))
    segment_map['bd'] = set(digits[4][0]) - segment_map['cf']

    segment_map['abcdf'] = segment_map['cf'].union(segment_map['bd']).union(segment_map['a'])

    segment_map['eg'] = set(digits[7][0]) - segment_map['abcdf']

    for c in digits[6]:
        diff = set(c) - segment_map['abcdf']
        if len(diff) == 1:
            segment_map['g'] = diff
    segment_map['e'] = segment_map['eg'] - segment_map['g']

    for c in digits[5]:
        u = set(c) - segment_map['a'].union(segment_map['g']).union(segment_map['bd'])
        if len(u) == 1:
            segment_map['f'] = u
    segment_map['c'] = segment_map['cf'] - segment_map['f']

    segment_map['acefg'] = segment_map['cf'].union(segment_map['eg']).union(segment_map['a'])
    for c in digits[6]:
        diff = set(c) - segment_map['acefg']
        if len(diff) == 1:
            segment_map['b'] = diff
    segment_map['d'] = segment_map['bd'] - segment_map['b']

    return {segment_map[s].pop(): s for s in segment_map.keys() if len(s) == 1}


digit_map = {'cf': '1',
             'acf': '7',
             'bcdf': '4',
             'acdeg': '2',
             'acdfg': '3',
             'abdfg': '5',
             'abdefg': '6',
             'abcefg': '0',
             'abcdfg': '9',
             'abcdefg': '8'}


def get_display_number(data):
    s_m = get_segment_map(data[0])
    num = []
    for s in data[1]:
        num.append(digit_map[''.join(sorted([s_m[c] for c in s]))])
    return int(''.join(num))


class Day8Test(unittest.TestCase):

    data = ['be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe',
            'edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc',
            'fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef | cg cg fdcagb cbg',
            'fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega | efabcd cedba gadfec cb',
            'aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga | gecf egdcabf bgf bfgea',
            'fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf | gebdcfa ecba ca fadegcb',
            'dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf | cefg dcbef fcge gbcadfe',
            'bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd | ed bcgafe cdgba cbgef',
            'egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg | gbdfcae bgc cg cgb',
            'gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc | fgae cfgab fg bagce']

    displays = get_display_numbers(data)

    def test_count_digits(self):
        data = get_display_numbers(self.data)
        self.assertEqual(26, count_output_digits(data))

    def test_get_segment_map(self):
        s_m = get_segment_map(self.displays[0][0])
        self.assertEqual(s_m['d'], 'a')
        self.assertEqual(s_m['g'], 'b')
        self.assertEqual(s_m['b'], 'c')
        self.assertEqual(s_m['c'], 'd')
        self.assertEqual(s_m['a'], 'e')
        self.assertEqual(s_m['e'], 'f')
        self.assertEqual(s_m['f'], 'g')


    def test_get_display_number(self):
        data = get_display_numbers(['acedgfb cdfbe gcdfa fbcad dab cefabd cdfgeb eafb cagedb ab | cdfeb fcadb cdfeb cdbaf'])
        self.assertEqual(5353, get_display_number(data[0]))

        self.assertEqual(61229, sum([get_display_number(d) for d in self.displays]))


def day8():
    file = open('day8input.txt')
    data = get_display_numbers(file)

    start_time = time.time()

    task1 = count_output_digits(data)
    task2 = sum([get_display_number(d) for d in data])

    return time.time() - start_time, task1, task2


# Day 9 - local minima & breadth first search

neighbours = [pt(-1, 0), pt(1, 0), pt(0, -1), pt(0, 1)]

def local_minima(m):
    minima = []

    for row in range(len(m)):
        for col in range(len(m[0])):
            x = m[row][col]
            min = True
            for n in neighbours:
                if len(m) > row + n.x >= 0 and len(m[0]) > col + n.y >= 0 and x >= m[row + n.x][col + n.y]:
                    min = False
                    break
            if min:
                minima.append(pt(row, col))
    return minima

def find_basin(p, m, seen, rows, cols):
    queue = deque()
    queue.append(p)

    seen[p] = 1
    size = 1

    while len(queue) > 0:
        s = queue.popleft()

        for n in neighbours:
            p_n = pt(s.x + n.x, s.y + n.y)
            if p_n.x in rows and p_n.y in cols and \
                    p_n not in seen and \
                    m[p_n.x][p_n.y] != 9:
                seen[p_n] = 1
                size += 1
                queue.append(p_n)

    return size


def find_basins(m):
    sizes = []
    seen = {}

    rows = range(len(m))
    cols = range(len(m[0]))

    for r in rows:
        for c in cols:
            p = pt(r, c)
            if m[r][c] != 9 and p not in seen:
                sizes.append(find_basin(p, m, seen, rows, cols))
    return sizes


class Day9Test(unittest.TestCase):
    data = ['2199943210',
            '3987894921',
            '9856789892',
            '8767896789',
            '9899965678']
    matrix = [[int(x) for x in list(row)] for row in data]

    def test_local_minima(self):
        self.assertEqual([1, 0, 5, 5], [self.matrix[p.x][p.y] for p in local_minima(self.matrix)])

    def test_find_basins(self):
        self.assertEqual([3, 9, 14, 9], find_basins(self.matrix))

def day9():
    data = [line.strip() for line in open('day9input.txt')]
    matrix = [[int(x) for x in list(row)] for row in data]
    start_time = time.time()

    minima = local_minima(matrix)
    task1 = sum([matrix[p.x][p.y] for p in minima]) + len(minima)

    sizes = find_basins(matrix)
    sizes.sort()
    task2 = reduce(operator.mul, sizes[-3:])

    return time.time() - start_time, task1, task2


# Day 10 - syntax checker
delims = {')': '(', '}': '{', '>': '<', ']': '['}

def parse(line):
    stack = []
    for c in line:
        if c in delims.keys():
            if len(stack) == 0 or delims[c] != stack.pop():
                return 'corrupted', c
        else:
            stack.append(c)

    return 'incomplete', stack

error_scores = {')': 3, '}': 1197, '>': 25137, ']': 57}
error_scores_incomplete = {'(': 1, '[': 2, '{': 3, '<': 4}

def score_incomplete(stacks):
    scores = []
    for stack in stacks:
        score2 = 0
        for d in reversed(stack):
            score2 = score2 * 5 + error_scores_incomplete[d]
        scores.append(score2)
    return scores

class Day10Test(unittest.TestCase):
    data = ['[({(<(())[]>[[{[]{<()<>>',
            '[(()[<>])]({[<{<<[]>>(',
            '{([(<{}[<>[]}>{[]{[(<()>',
            '(((({<>}<{<{<>}{[]{[]{}',
            '[[<[([]))<([[{}[[()]]]',
            '[{[{({}]{}}([{[{{{}}([]',
            '{<[[]]>}<{[{[{[]{()[[[]',
            '[<(<(<(<{}))><([]([]()',
            '<{([([[(<>()){}]>(<<{{',
            '<{([{{}}[<[[[<>{}]]]>[]]']

    def test_parse(self):
        self.assertEqual(8, len(parse(self.data[0])[1]))
        self.assertEqual('}', parse(self.data[2])[1])

    def test_score_incomplete(self):
        incomplete = []
        for d in self.data:
            code, res = parse(d)
            if code == 'incomplete':
                incomplete.append(res)
        self.assertEqual([288957, 5566, 1480781, 995444, 294], score_incomplete(incomplete))


def day10():
    data = [line.strip() for line in open('day10input.txt')]
    start_time = time.time()

    keys = ['incomplete']
    keys.extend(delims.keys())
    results = {k: 0 for k in  keys}

    incomplete = []

    for d in data:
        code, res = parse(d)
        if code == 'corrupted':
            results[res] += 1
        else:
            incomplete.append(res)

    score = 0
    for k, v in error_scores.items():
        score += v * results[k]
    task1 = score

    task2 = statistics.median(score_incomplete(incomplete))

    return time.time() - start_time, task1, task2


# Day 11 - Flashing dumbo octopie

def flash_neighbours(p, step1, size):
    flashed = {}
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            n = pt(p.x + i, p.y + j)
            if n.x in size and n.y in size and n != p:
                flashed[n] = step1[n.x][n.y] + 1 if n not in flashed else flashed[n] + 1

    return flashed

def flash(octopie):
    step1 = [[x + 1 for x in list(row)] for row in octopie]
    flashed = {}

    size = range(len(step1))

    for r in size:
        for c in size:
            if step1[r][c] > 9:
                new_n = flash_neighbours(pt(r, c), step1, size)
                for i in new_n:
                    flashed[i] = flashed[i] + new_n[i] - step1[i.x][i.y] if i in flashed else new_n[i]

    while sum([i > 9 for i in flashed.values()]):
        p = next(iter(flashed))
        n = flashed.pop(p)

        if n <= 9:
            flashed[p] = n
        elif step1[p.x][p.y] <= 9 and n > 9:
                step1[p.x][p.y] = 10
                new_n = flash_neighbours(p, step1, size)
                for i in new_n:
                    flashed[i] = flashed[i] + new_n[i] - step1[i.x][i.y] if i in flashed else new_n[i]

    res = [[0 for i in size] for j in size]
    for r in size:
        for c in size:
            p = pt(r, c)
            if p in flashed:
                res[r][c] = flashed[p]
            elif step1[r][c] < 10:
                res[r][c] = step1[r][c]

    return res

class Day11Test(unittest.TestCase):
    data = ['5483143223',
            '2745854711',
            '5264556173',
            '6141336146',
            '6357385478',
            '4167524645',
            '2176841721',
            '6882881134',
            '4846848554',
            '5283751526']
    matrix = [[int(x) for x in list(row)] for row in data]

    data2 = ['11111',
             '19991',
             '19191',
             '19991',
             '11111']
    matrix2 = [[int(x) for x in list(row)] for row in data2]

    def test_flash_neighbours(self):
        exp = {p: 2 for p in [pt(0, 0), pt(0, 1), pt(0, 2), pt(1, 0), pt(1, 2), pt(2, 0), pt(2, 2)]}
        exp[pt(1, 2)] = 10
        exp[pt(2, 1)] = 10
        self.assertEqual(exp, flash_neighbours(pt(1, 1), self.matrix2, range(5)))

    def test_flash(self):
        exp = [[int(x) for x in list(row)] for row in
               ['34543',
                '40004',
                '50005',
                '40004',
                '34543']]
        self.assertEqual(exp, flash(self.matrix2))

    def test_big_flash(self):
        exp = [[int(x) for x in list(row)] for row in
               ['6594254334',
                '3856965822',
                '6375667284',
                '7252447257',
                '7468496589',
                '5278635756',
                '3287952832',
                '7993992245',
                '5957959665',
                '6394862637']]
        actual = flash(self.matrix)
        self.assertEqual(exp, actual)

        exp2 = [[int(x) for x in list(row)] for row in
               ['8807476555',
                '5089087054',
                '8597889608',
                '8485769600',
                '8700908800',
                '6600088989',
                '6800005943',
                '0000007456',
                '9000000876',
                '8700006848']]
        actual = flash(actual)
        self.assertEqual(exp2, actual)

        for i in range(8):
            actual = flash(actual)

        exp10 = [[int(x) for x in list(row)] for row in
               ['0481112976',
                '0031112009',
                '0041112504',
                '0081111406',
                '0099111306',
                '0093511233',
                '0442361130',
                '5532252350',
                '0532250600',
                '0032240000']]
        self.assertEqual(exp10, actual)




def day11():
    data = [line.strip() for line in open('day11input.txt')]
    octopie = [[int(x) for x in list(row)] for row in data]

    start_time = time.time()

    no_flashes = 0
    for i in range(100):
        octopie = flash(octopie)
        zeros = [row.count(0) for row in octopie]
        no_flashes += sum(zeros)

    task1 = no_flashes

    i = 100
    zeros = []
    while sum(zeros) < 100:
        octopie = flash(octopie)
        zeros = [row.count(0) for row in octopie]
        i += 1

    task2 = i

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
    for i in range(1, 12):
        run(eval("day" + str(i)))
