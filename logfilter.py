class OutputFilter(logging.Filter):
    def __init__(self, keywords, name=None):
        super(OutputFilter, self).__init__(name)
        self.keywords = keywords

    def filter(self, record):
        msg = record.getMessage()
        return not any(k in msg for k in self.keywords)
