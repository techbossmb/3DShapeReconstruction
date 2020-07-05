import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

def __load_voxelgrid_data(filepath):
    data = json.loads(open(filepath).read())
    voxels = data['voxels']
    voxel_size = len(voxels)
    x = np.zeros(voxel_size)
    y = np.zeros(voxel_size)
    z = np.zeros(voxel_size)
    for i, voxel in enumerate(voxels):
        x[i] = voxel['x']
        y[i] = voxel['y']
        z[i] = voxel['z']
    x = x.astype(int)
    y = y.astype(int)
    z = z.astype(int)
    return x,y,z

def __generate_subsamples(data, subsample_ratio=0.1, n_samples=100, shape=16):
    x,y,z = data
    input_pointclouds = np.zeros((n_samples, shape, shape, shape))
    for i in range(n_samples):
        subsampled_ix = np.random.choice(len(x), int(subsample_ratio*len(x)), replace=False)
        x_sub = x[subsampled_ix]
        y_sub = y[subsampled_ix]
        z_sub = z[subsampled_ix]
        for k in range(len(x_sub)):
            input_pointclouds[i][x_sub[k]][y_sub[k]][z_sub[k]] = 1
    input_pointclouds = input_pointclouds.astype(int)
    input_pointclouds = input_pointclouds.reshape(n_samples, shape, shape, shape,1)
    return input_pointclouds

def __voxel_to_pointcloud(data, n_samples=100, shape=16):
    x,y,z = data
    output_pointclouds = np.zeros((n_samples,shape,shape,shape))
    for i in range(n_samples):
        for j in range(len(x)):
            output_pointclouds[i][x[j]][y[j]][z[j]] = 1
    output_pointclouds = output_pointclouds.astype(int)
    output_pointclouds = output_pointclouds.reshape(n_samples, shape, shape, shape,1)
    return output_pointclouds

def get_shapes_data():
    shape = 16
    shape_files = ['cube', 'sphere', 'torus']
    n_samples_per_shape = 1000
    input_data = np.zeros((n_samples_per_shape*len(shape_files), shape, shape, shape, 1))
    output_data = np.zeros((n_samples_per_shape*len(shape_files), shape, shape, shape, 1))
    for i,shape_file in enumerate(shape_files):
        filepath = '..{0}data{0}{1}16.json'.format(os.sep, shape_file)
        print('loading {}'.format('..{0}data{0}{1}16.json'.format(os.sep, shape_file)))
        x,y,z = __load_voxelgrid_data(filepath)
        input_pointclouds = __generate_subsamples((x,y,z), n_samples = n_samples_per_shape)
        output_pointclouds = __voxel_to_pointcloud((x,y,z), n_samples = n_samples_per_shape)
        input_data[i*n_samples_per_shape : (i+1)*n_samples_per_shape] = input_pointclouds
        output_data[i*n_samples_per_shape : (i+1)*n_samples_per_shape] = output_pointclouds
    return input_data, output_data

def get_train_test_data():
    print('loading train-test data')
    x, y = get_shapes_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test

