import time
import numpy as np


class Timer:
    """
    Define a timer class to record various running times
    """
    def __init__(self):
        self.times = []
        self.start()

    # Start recording time
    def start(self):
        self.tik = time.time()

    # Stop recording time and store the elapsed time in a list
    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    # Return the average duration of consumption
    def avg(self):
        return sum(self.times) / len(self.times)

    # Return the total running time of the model
    def sum(self):
        return sum(self.times)

    # Return the accumulated time
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()