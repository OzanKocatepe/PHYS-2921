class EventCounter:
    """Tracks the number of events we have parsed through.
    
    It is useful to group ParticleEvents based on which <event> group they
    occurred in. These groups are not labelled in the files, and we need to
    have unique group labels between files.
    
    We could use random values to statistically ensure the labels are different,
    but this is slower than just incrementing a counter. However, since different files
    have individual function calls to parse through them, we use a seperate class
    with a static variable to count the events.
    
    Attributes:
        currentEvent: int
            Keeps track of which event group we are currently parsing through.
            
    Methods:
        IncrementCounter:
            Increments the currentEvent variable.
    """

    currentEvent: int = 0

    def __init__(self):
        pass

    def IncrementCounter() -> None:
        """Increments the counter variable.
        """
        EventCounter.currentEvent += 1