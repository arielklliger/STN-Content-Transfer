#taken from https://medium.com/@shanlins/spatial-transformer-networks-stn-and-its-implementation-2638d58d41f8

import torch.nn as nn
import torch.nn.functional as F
import keras as K

# Spatial transformer network forward function
def stn(self, x):
    xs = self.localization(x)
    xs = xs.view(-1, 10 * 3 * 3)
    theta = self.fc_loc(xs)
    theta = theta.view(-1, 2, 3)

    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid)

    return x


def _transform(self, X, affine_transformation, output_size):
    batch_size, num_channels = K.shape(X)[0], K.shape(X)[3]
    transformations = K.reshape(affine_transformation,
                                shape=(batch_size, 2, 3))
    regular_grids = self._make_regular_grids(batch_size,
                                             *output_size)
    sampled_grids = K.batch_dot(transformations, regular_grids)
    interpolated_image = self._interpolate(X, sampled_grids,
                                           output_size)
    new_shape = (batch_size, output_size[0],
                 output_size[1], num_channels)
    interpolated_image = K.reshape(interpolated_image, new_shape)

    return interpolated_image


def _make_regular_grids(self, batch_size, height, width):
    # making a single regular grid
    x_linspace = K_linspace(-1., 1., width)
    y_linspace = K_linspace(-1., 1., height)
    x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
    x_coordinates = K.flatten(x_coordinates)
    y_coordinates = K.flatten(y_coordinates)
    ones = K.ones_like(x_coordinates)
    grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)
    # repeating grids for each batch
    grid = K.flatten(grid)
    grids = K.tile(grid, K.stack([batch_size]))

    return K.reshape(grids, (batch_size, 3, height * width))




def _interpolate(self, image, sampled_grids, output_size):
    batch_size = K.shape(image)[0]
    height = K.shape(image)[1]
    width = K.shape(image)[2]
    num_channels = K.shape(image)[3]
    x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
    y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')


    #x = .5 * (x + 1.0) * K.cast(height-1, dtype='float32')
    #y = .5 * (y + 1.0) * K.cast(width-1, dtype='float32')

    # x0 = K.cast(x, 'int32')
    # x1 = x0 + 1
    # y0 = K.cast(y, 'int32')
    # y1 = y0 + 1
    # max_x = int(K.int_shape(image)[1] - 1)
    # max_y = int(K.int_shape(image)[2] - 1)
    # x0 = K.clip(x0, 0, max_x)
    # x1 = K.clip(x1, 0, max_x)
    # y0 = K.clip(y0, 0, max_y)
    # y1 = K.clip(y1, 0, max_y)

    pixels_batch = K.arange(0, batch_size) * (height * width)
    pixels_batch = K.expand_dims(pixels_batch, axis=-1)
    flat_output_size = output_size[0] * output_size[1]
    base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
    base = K.flatten(base)

    base_y0 = y0 * width
    base_y0 = base + base_y0
    base_y1 = y1 * width
    base_y1 = base_y1 + base
    indices_a = base_y0 + x0
    indices_b = base_y1 + x0
    indices_c = base_y0 + x1
    indices_d = base_y1 + x

    flat_image = K.reshape(image, shape=(-1, num_channels))
    flat_image = K.cast(flat_image, dtype='float32')
    pixel_values_a = K.gather(flat_image, indices_a)
    pixel_values_b = K.gather(flat_image, indices_b)
    pixel_values_c = K.gather(flat_image, indices_c)
    pixel_values_d = K.gather(flat_image, indices_d)

    x0 = K.cast(x0, 'float32')
    x1 = K.cast(x1, 'float32')
    y0 = K.cast(y0, 'float32')
    y1 = K.cast(y1, 'float32')
    area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
    area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
    area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
    area_d = K.expand_dims(((x - x0) * (y - y0)), 1)
    values_a = area_a * pixel_values_a
    values_b = area_b * pixel_values_b
    values_c = area_c * pixel_values_c
    values_d = area_d * pixel_values_d

    return values_a + values_b + values_c + values_d


