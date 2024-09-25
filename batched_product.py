from itertools import product, islice

def batched_product(list1, list2, batch_size=100000):
    iterator = product(list1, list2)
    ans = list(islice(iterator, batch_size))
    while len(ans) > 0:
        yield ans
        ans = list(islice(iterator, batch_size))


"""
class batched_product(object):
    def __init__(self, list1, list2, batch_size=1000000):
        self.list1 = list1
        self.list2 = list2
        self.len1 = len(list1)
        self.len2 = len(list2)
        self.ind1 = 0
        self.batch_size = batch_size
        self.cur_batch_size = 0
        self.num_batches = 0

    def __iter__(self):
        return self

    def __next__(self): #
        ans = []
        if self.ind1 >= self.len1:
            raise StopIteration
        while (self.cur_batch_size + self.len2 <= self.batch_size) and (self.ind1 < self.len1):
            ans.extend(list(product([self.list1[self.ind1]], self.list2)))
            self.cur_batch_size += self.len2
            self.ind1 += 1
        if self.ind1 == 0:
            raise "Overfull"
        self.num_batches += 1
        self.cur_batch_size = 0
        return ans
"""
