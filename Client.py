import socket


def Client():
    host = '192.168.1.111'  # client ip
    port = 4005

    server = ('192.168.1.117', 4000)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))

    message = input("-> ")
    while message != 'q':
        s.sendto(message.encode('utf-8'), server)
        message = input("-> ")
    s.close()


if __name__ == '__main__':
    Client()