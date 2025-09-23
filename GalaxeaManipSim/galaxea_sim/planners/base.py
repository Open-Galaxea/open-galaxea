class BasePlanner:
    def __init__(self):
        pass
    
    def solve(self, substep, *args, **kwargs):
        raise NotImplementedError