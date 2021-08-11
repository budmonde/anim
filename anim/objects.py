import numpy as np
from skimage.filters import gabor_kernel


def gaussian(inp, mean, sigma):
    return np.exp(-((inp - mean) / sigma) ** 2 / 2)
#  / ((2 * np.pi) ** 0.5 * sigma)


class GaussianBlob:
    def __init__(self, location, peak_luminance, sigma):
        self.location = location
        self.peak_luminance = peak_luminance
        self.sigma = sigma

    def draw(self, arr):
        x = np.linspace(-1, 1, arr.shape[2])
        y = np.linspace(-1, 1, arr.shape[1])

        gauss_x = gaussian(x, self.location[0], self.sigma)
        gauss_y = gaussian(y, self.location[1], self.sigma)
        gauss = (gauss_x[None, :] * gauss_y[:, None])

        arr += gauss * self.peak_luminance


class GaborBlob:
    def __init__(self, location, peak_luminance, freq, sigma):
        self.location = location
        self.peak_luminance = peak_luminance
        self.freq = freq
        self.sigma = sigma

    def draw(self, arr):
        x = np.linspace(-1, 1, arr.shape[2])
        y = np.linspace(-1, 1, arr.shape[1])

        gauss_x = gaussian(x, self.location[0], self.sigma)
        gauss_y = gaussian(y, self.location[1], self.sigma)
        gauss = (gauss_x[None, :] * gauss_y[:, None])

        sine = np.cos(2 * np.pi * self.freq * x)

        field = gauss * sine

        arr += field * self.peak_luminance


class TranslatingSine:
    def __init__(self, location, angle, amplitude, size, wavelength, velocity):
        self.location = location
        self.angle = angle
        self.amplitude = amplitude
        self.size = size
        self.wavelength = wavelength
        self.velocity = velocity
        self.t = 0

    def draw(self, target):
        x = np.arange(self.size)
        y = np.arange(self.size)
        xv, yv = np.meshgrid(x, y)
        if self.angle == 0:
            patch = self.amplitude * np.sin(2 * np.pi / self.wavelength * (xv - self.velocity * self.t))
        elif self.angle == 90:
            patch = self.amplitude * np.sin(2 * np.pi / self.wavelength * (yv - self.velocity * self.t))
        patch = (patch * 0.5) + 0.5

        offset = (np.array(target.shape)[1:] - 1) * self.location
        offset = np.round(offset).astype(np.int32)
        offset -= np.array(patch.shape) // 2

        target[
            :,
            offset[0] : offset[0] + patch.shape[0],
            offset[1] : offset[1] + patch.shape[1],
        ] = patch

    def update(self, dt):
        self.t += dt


class BlinkingSquare:
    def __init__(self, location, size, freq):
        self.location = location
        self.size = size
        self.freq = freq
        self.t = 0

    def draw(self, target):
        patch = np.ones((self.size, self.size))
        patch *= np.cos(2 * np.pi * self.freq * self.t)
        patch_shape = np.array(patch.shape)

        offset = (
            np.round((np.array(target.shape[1:]) - 1) * self.location).astype(np.int32)
            - patch_shape // 2
        )

        target[
            :,
            offset[0] : offset[0] + patch.shape[0],
            offset[1] : offset[1] + patch.shape[1],
        ] = patch

    def update(self, dt):
        self.t += dt


class BlinkingGabor:
    def __init__(self, location, freq):
        self.location = location
        self.freq = freq
        self.t = 0

    def draw(self, target):
        kernel = np.real(gabor_kernel(0.1, theta=0, sigma_x=10, sigma_y=10))
        kernel /= max(abs(kernel.min()), abs(kernel.max()))
        kernel *= np.cos(2 * np.pi * self.freq * self.t)
        kernel += 0.5
        kernel_shape = np.array(kernel.shape)

        offset = np.int32(np.array(target.shape[1:]) * self.location) - kernel_shape // 2
        target[
            :,
            offset[0] : offset[0] + kernel.shape[0],
            offset[1] : offset[1] + kernel.shape[1],
        ] = kernel

    def update(self, dt):
        self.t += dt


class FreezeFrameObject:
    def __init__(self, obj, start_time):
        self.obj = obj
        self.start_time = start_time
        self.t = 0

    def draw(self, canvas):
        self.obj.draw(canvas)

    def update(self, dt):
        if self.t >= self.start_time:
            self.obj.update(dt)
        self.t += dt
