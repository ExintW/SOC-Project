class TeeOutput:
    def __init__(self, file, original_stdout):
        self.file = file
        self.original_stdout = original_stdout
    
    def write(self, text):
        self.file.write(text)
        if ("Processing year" in text or 
            "=======================================================================" in text or
            "Year" in text and "Month" in text or
            "Completed simulation for Year" in text):
            self.original_stdout.write(text)
        self.file.flush()
    
    def flush(self):
        self.file.flush()
        self.original_stdout.flush()