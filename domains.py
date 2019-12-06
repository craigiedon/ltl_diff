import torch

class Box:
    """ A domain defined by a bounding box with min point a and max point b"""
    def __init__(self, a, b):
        self.a = a
        self.b = b
        assert not self.is_empty()

    def is_empty(self):
        return torch.any(self.a > self.b)
        
    def project(self, x):
        return torch.min(torch.max(x, self.a), self.b)

    def sample(self):
        s = self.a + (self.b - self.a) * torch.rand_like(self.a)
        return s

class Segment:
    """ A domain defined by a line segment with start/end points p1/p2 """
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.dir = self.p2 - self.p1
        self.dist = self.dir.norm()
        self.unit_dir = self.dir / self.dir.norm()

    def is_empty(self):
        # A line segment can never be empty
        return False

    def project(self, x):
        # dot product
        dot_product = torch.sum((x - self.p1) * self.unit_dir)
        if dot_product < 0:
            return self.p1
        if dot_product > self.dist:
            return self.p2

        return self.p1_n + dot_product * self.unit_dir

    def sample(self):
        return self.p1 + (self.p2 - self.p1) * torch.rand()