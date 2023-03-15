from collections import OrderedDict


class VehicleCache:

    def __init__(self):
        self.vehicles = OrderedDict()
        self.undetected = OrderedDict()
        self.nextId = 0

    def add(self, vehicle):
        self.vehicles[self.nextId] = vehicle
        self.undetected[self.nextId] = 0
        self.nextId += 1

    def remove(self, vehicle_id):
        del self.vehicles[vehicle_id]
        del self.undetected[vehicle_id]

