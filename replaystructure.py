from random import choices


class Experience:
    
    '''
    An experience to store in the replay buffers
    
    '''
    
    def __init__(self,state,action,reward,new_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
    
class ReplayBuffer:
    
    '''
    A buffer to store past experiences
    '''

    def __init__(self, buffer_size = 1000):
        '''
        self.buffer is a list of the most recent experiences
        self.prioritity is a list of priorities for the experiences to sample experiences 
            which are more "interesting" (Prioritized Experience Replay)
        self.buffer_size is the number of previous experiences stored in the buffer to limit memory
        
        '''
        
        self.buffer = []
        self.priority = []
        self.buffer_size = buffer_size
        
    
    
    def add_to_buffer(self,experience, priority = 1):
        '''
        Adds an experience to the buffer
        
        '''
        
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
            self.priority.pop(0)
        self.buffer.append(experience)
        self.priority.append(priority)
        
    def sample(self,n):
        '''
        Returns n experiences taking into consideration the priorities
        
        '''
        
        return choices(self.buffer, self.priority,k=n)
        