import doctest

def doctest_test(a):
    """
    >>> doctest_test(1)[0]
    1
    >>> doctest_test(20)[0]
    20
    """
    score = 50
    # insert your solution from here..

    return a, score


if __name__ == '__main__':
    doctest.testmod()