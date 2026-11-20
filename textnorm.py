class TextNorm:

    def __init__(self,Data):
        self.Data = Data

    def transform_text(self,Data):
        text = []
        for sentence in Data:
            token = sentence.lower().split()
            text.append(token)
        return text

    