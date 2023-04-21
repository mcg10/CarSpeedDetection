import socket


class Client:
    def __init__(self):
        host = 'CLIENT IP'  # client ip
        port = 4005

        self.server = ('SERVER IP', 4000)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((host, port))

    def send(self, vehicle_id):
        message = 'Vehicle {} was speeding'.format(vehicle_id)
        self.s.sendto(message.encode('utf-8'), self.server)

    def shut_down(self):
        self.s.close()


if __name__ == '__main__':
    Client()
