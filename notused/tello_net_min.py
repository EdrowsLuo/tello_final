
class TelloRequest:

    def __init__(self, telloServer):
        self.telloServer = telloServer
        self.body = ""

    def setBody(self, command):
        self.body = command



class TelloServer:

    def __init__(self):
        pass



