import numpy as np


class Canvas:
    """
        base_luminance: base luminance of the canvas in cd/m^2
    """
    def __init__(self, resolution, base_luminance=50.0):
        self.resolution = resolution
        self.base_luminance = base_luminance
        self.objects = dict()
        self.max_z = 0
        self.t = 0

    def add(self, obj, z=-1):
        if z == -1:
            z = self.max_z + 1
        if z > self.max_z:
            self.max_z = z
        self.objects[obj] = z

    def render(self, order='CHW'):
        arr = np.ones((1, *self.resolution)) * self.base_luminance
        for obj in [k for k, _ in sorted(self.objects.items(), key=lambda x: x[1])]:
            obj.draw(arr)

        if order == "CHW":
            return arr
        elif order == "HWC":
            return np.transpose(arr, (1, 2, 0))
        else:
            raise NotImplementedError("Channel order not supported.")

    def update(self, dt):
        for obj in self.objects:
            obj.update(dt)
        self.t += dt
