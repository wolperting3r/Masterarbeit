import numpy as np
from sklearn.base import (
    BaseEstimator,  # for get_params, set_params
    TransformerMixin     # for fit_transform
)


class TransformData(BaseEstimator, TransformerMixin):
    def __init__(self, parameters, reshape=False):
        self.parameters = parameters
        self.reshape = reshape

    def fit(self, dataset):
        return self  # Nothing to do

    def transform(self, dataset):
        # Split the training and test data into labels (first column) and data
        labels = np.round(dataset.iloc[:, 0].to_numpy(), 3)
        data = np.round(dataset.iloc[:, 1:].to_numpy(), 3)

        if self.reshape:
            st_sz = self.parameters['stencil_size']
            # Reshape data
            data = np.reshape(data, (data.shape[0], st_sz[0], st_sz[1], 1))

        return [labels, data]


class FindGradient(BaseEstimator, TransformerMixin):
    def __init__(self, parameters):
        self.parameters = parameters

    def fit(self, dataset):
        return self

    def transform(self, dataset):
        # Get stencil size
        st_sz = self.parameters['stencil_size']
        # Seperate data from labels
        data = dataset[1].copy()
        # Get shape of data
        shape = data.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            data = np.reshape(data, (shape[0], st_sz[0], st_sz[1], 1))
        # Get shape again
        shape = data.shape
        # Calculate midpoint
        mid_point = [int((shape[1]-1)/2), int((shape[2]-1)/2)]
        # Calculate gradient vectors around midpoint
        gradient = np.array([
            data[0:shape[0]-1, mid_point[0]+1, mid_point[1], shape[3]-1] -
            data[0:shape[0]-1, mid_point[0]-1, mid_point[1], shape[3]-1],  # positiv x
            data[0:shape[0]-1, mid_point[0], mid_point[1]+1, shape[3]-1] -
            data[0:shape[0]-1, mid_point[0], mid_point[1]-1, shape[3]-1]  # negative y
        ])
        # Calculate angles of gradient vectors
        angle = np.arctan2(-gradient[1], gradient[0])

        return [dataset[0], dataset[1], angle]


class Rotate(BaseEstimator, TransformerMixin):
    def __init__(self, parameters):
        self.parameters = parameters

    def fit(self, dataset):
        return self

    def transform(self, dataset):
        # Get stencil size
        st_sz = self.parameters['stencil_size']
        # Seperate dataset
        data = dataset[1].copy()
        angle = dataset[2].copy()
        # Get shape of data
        shape = data.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            data = np.reshape(data, (shape[0], st_sz[0], st_sz[1], 1))
        # Calculate rotation based on angle
        rotation = (np.floor((angle + np.pi)*2/np.pi)).astype(int)
        # Get indices where vof values should be rotated 0/90/180/270
        rotation0 = np.argwhere(rotation == 0)
        rotation90 = np.argwhere(rotation == 1)
        rotation180 = np.argwhere(rotation == 2)
        rotation270 = np.argwhere(rotation == 3)
        # Rotate
        data[rotation0] = data[rotation0]
        data[rotation90] = np.rot90(data[rotation90], 1, axes=(2, 3))
        data[rotation180] = np.rot90(data[rotation180], 2, axes=(2, 3))
        data[rotation270] = np.rot90(data[rotation270], 3, axes=(2, 3))
        if data.shape != shape:
            # Reshape rotated data to original shape
            data = np.reshape(data, shape)

        '''
        # Test
        ind = 6
        print(f'\ngradient[:,{ind}]: {gradient[:,ind]}')
        print(f'angle[{ind}]: {angle[ind]*180/np.pi}')
        print(f'rotation[{ind}]: {rotation[ind]}')
        # '''
        return [dataset[0], data, dataset[2]]
