import json


class Recorder:
    """
    Recorder class to recording things like trajectories
    """

    def __init__(self):
        self.data = []
        self.n = 0

    def rec(self, new_data):
        self.data.append(new_data)
        self.n += 1
        return self

    def save(self, file):
        with open(file, 'w') as f:
            json.dump(self.data, f)
        return self

    def load(self, file):
        with open(file, 'r') as f:
            self.data = json.load(f)
            self.n = len(self.data)
        return self

    def sum(self, fun, disc=1.0, start=0, end=None):
        if end is None:
            end = self.n
        s = 0
        for idx in reversed(range(start, end)):
            s *= disc
            s += fun(self.data[idx])
        return s


if __name__ == '__main__':

    def my_fun(x):
        return x ** 2


    n = 10
    recorder = Recorder()
    for _ in range(n):
        recorder.rec(1)

    print(recorder.data)
    print("-" * 20)
    print(recorder.sum(my_fun, disc=1))
    print("-" * 20)
    print(recorder.sum(my_fun, disc=0.8))
