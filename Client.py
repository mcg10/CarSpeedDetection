import socket


class Client:
    def __init__(self):
        host = '192.168.1.111'  # client ip
        port = 4005

        self.server = ('192.168.1.117', 4000)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((host, port))

    def send(self, vehicle_id):
        message = 'Vehicle {} was speeding'.format(vehicle_id)
        self.s.sendto(message.encode('utf-8'), self.server)

    def shut_down(self):
        self.s.close()


if __name__ == '__main__':
    Client()
