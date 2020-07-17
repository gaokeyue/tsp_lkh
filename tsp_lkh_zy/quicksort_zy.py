import random


def quicksort(seq, p: int = 0, r=None, reverse=False):
    if r is None:
        r = len(seq)-1
    if p < r:
        q = partition(seq, p, r, reverse)
        quicksort(seq, p, q-1, reverse)
        quicksort(seq, q+1, r, reverse)
    return


def partition(seq, p, r, reverse):
    x = seq[r]
    i = p-1
    if reverse:
        for j in range(p, r):
            if seq[j] >= x:
                i += 1
                seq[i], seq[j] = seq[j], seq[i]
    else:
        for j in range(p, r):
            if seq[j] <= x:
                i += 1
                seq[i], seq[j] = seq[j], seq[i]
    seq[i+1], seq[r] = seq[r], seq[i+1]
    return i+1


def randomized_partition(seq, p, r, reverse):
    i = random.randint(p, r)
    seq[i], seq[r] = seq[r], seq[i]
    q = partition(seq, p, r, reverse)
    return q


def randomized_quicksort(seq, p: int = 0, r=None, reverse=False):
    if r is None:
        r = len(seq)-1
    if p < r:
        q = randomized_partition(seq, p, r, reverse)
        randomized_quicksort(seq, p, q-1, reverse)
        randomized_quicksort(seq, q+1, r, reverse)
    return


if __name__ == '__main__':
    test_list = [1, 9, 9, 4, 0, 6, 2, 0]
    quicksort(test_list)
    print(test_list)
    randomized_quicksort(test_list, reverse=True)
    print(test_list)
