import random

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def bright_color():
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

def pastel_color():
    return tuple(min(255, int(c * 0.7 + 255 * 0.3)) for c in random_color())
