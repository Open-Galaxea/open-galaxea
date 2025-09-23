class BaseController:
    def __init__(self):
        pass
    
    def get_control_signal(self, action):
        raise NotImplementedError()
    
    @property
    def action_dim(self):
        raise NotImplementedError()
    
    @property
    def action_space(self):
        raise NotImplementedError()
    
    def reset(self):
        pass