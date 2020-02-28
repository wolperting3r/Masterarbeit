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

    def get_gradient(self, data):
        # Get points where gradient should not be calculated
        no_interface = np.nonzero((data == 1) | (data == 0))
        # Initialize gradient matrices
        grad_x = np.empty((data.shape)) * np.nan
        grad_y = np.empty((data.shape)) * np.nan
        # If stencil is 5x5 or bigger, use central difference quotient
        if (data.shape[1] >= 5) & (data.shape[2] >= 5):
            # Calculate gradient
            for x in range(1, data.shape[1]-1):
                for y in range(1, data.shape[2]-1):
                    grad_x[:, y, x, 0] = np.array([
                        data[:, y, x+1, 0] -
                        data[:, y, x-1, 0]
                    ])/2
                    grad_y[:, y, x, 0] = np.array([
                        data[:, y-1, x, 0] -
                        data[:, y+1, x, 0]
                    ])/2
        # If stencil has other dimensions (e.g. 3x3), use fallback (forward/central/backward d.q.)
        else:
            mp = [int((data.shape[1]-1)/2), int((data.shape[2]-1)/2)]  # y, x
            # Calculate gradient x
            for y in range(mp[0]-1, mp[0]+2): 
                # Backward difference quotient
                grad_x[:, y, mp[1]-1, 0] = np.array([
                    data[:, y, mp[1], 0] -
                    data[:, y, mp[1]-1, 0]
                ])/1  # mathematically correct: /1, but /2 works much better in this application
                # Central difference quotient
                grad_x[:, y, mp[1], 0] = np.array([
                    data[:, y, mp[1]+1, 0] -
                    data[:, y, mp[1]-1, 0]
                ])/2
                # Forward difference quotient
                grad_x[:, y, mp[1]+1, 0] = np.array([
                    data[:, y, mp[1]+1, 0] -
                    data[:, y, mp[1], 0]
                ])/1
            for x in range(mp[1]-1, mp[1]+2): 
                # Backward difference quotient
                grad_y[:, mp[0]-1, x, 0] = np.array([
                    data[:, mp[0]-1, x, 0] -
                    data[:, mp[0], x, 0]
                ])/1
                # Central difference quotient
                grad_y[:, mp[0], x, 0] = np.array([
                    data[:, mp[0]-1, x, 0] -
                    data[:, mp[0]+1, x, 0]
                ])/2
                # Forward difference quotient
                grad_y[:, mp[0]+1, x, 0] = np.array([
                    data[:, mp[0], x, 0] -
                    data[:, mp[0]+1, x, 0]
                ])/1
        grad_x[no_interface] = np.nan
        grad_y[no_interface] = np.nan
        # angle = np.arctan2(-grad_y, -grad_x)*180/np.pi  # y, x
        # angle = gradient[1]
        return [grad_y, grad_x]

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
        # Get gradient matrix
        [grad_y, grad_x] = self.get_gradient(data)

        '''
        # Test
        ind = 1
        print_data_gradx = grad_x.transpose((0, 1, 3, 2))[ind]
        print_data_grady = grad_y.transpose((0, 1, 3, 2))[ind]
        print_data_grad = data.transpose((0, 1, 3, 2))[ind]
        print(f'\nGrad_x:\n{print_data_gradx}')
        print(f'\nGrad_y:\n{print_data_grady}')
        print(f'\nData:\n{print_data_grad}')
        # '''

        if data.shape != shape:
            # Reshape transformed data to original shape
            grad_y = np.reshape(grad_y, shape)
            grad_x = np.reshape(grad_x, shape)

        return [dataset[0], dataset[1], grad_x, grad_y]


class FindKappa(BaseEstimator, TransformerMixin):
    def __init__(self, parameters):
        self.parameters = parameters

    def fit(self, dataset):
        return self
    
    def transform(self, dataset):
        '''
        NOT WORKING YET
        '''
        # Seperate dataset
        grad_x = dataset[2].copy()
        grad_y = dataset[3].copy()

        # Get stencil size
        st_sz = self.parameters['stencil_size']
        # Get shape of data
        shape = grad_x.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            grad_x = np.reshape(grad_x, (shape[0], st_sz[0], st_sz[1], 1))
            grad_y = np.reshape(grad_y, (shape[0], st_sz[0], st_sz[1], 1))

        # Calculate midpoint
        mp = [int((st_sz[0]-1)/2), int((st_sz[1]-1)/2)]

        # Calculate angle
        angle_matrix = np.arctan2(-grad_y, -grad_x)
        dnxdx1 = angle_matrix[:, mp[0],   mp[1]+1, 0] - angle_matrix[:, mp[0],   mp[1]-1, 0]
        dnydy1 = angle_matrix[:, mp[0]+1, mp[1],   0] - angle_matrix[:, mp[0]-1, mp[1],   0]
        dnxdx2 = angle_matrix[:, mp[0]+1, mp[1]+1, 0] - angle_matrix[:, mp[0]-1, mp[1]-1, 0]
        dnydy2 = angle_matrix[:, mp[0]+1, mp[1]-1, 0] - angle_matrix[:, mp[0]-1, mp[1]+1, 0]
        kappa = dnxdx1-dnydy1+dnxdx2-dnydy2

        '''
        # Calculate normal vector
        # n_x = grad_x/(np.sqrt(grad_x*grad_x + grad_y*grad_y))
        # n_y = grad_y/(np.sqrt(grad_x*grad_x + grad_y*grad_y))
        n_x = grad_x
        n_y = grad_y
        # Calculate dn_xi/dxi
        dnxdx1 = n_x[:, mp[0],   mp[1]+1, 0] - n_x[:, mp[0],   mp[1]-1, 0]
        dnydy1 = n_y[:, mp[0]+1, mp[1],   0] - n_x[:, mp[0]-1, mp[1],   0]
        dnxdx2 = n_x[:, mp[0]+1, mp[1]+1, 0] - n_x[:, mp[0]-1, mp[1]-1, 0]
        dnydy2 = n_y[:, mp[0]+1, mp[1]-1, 0] - n_x[:, mp[0]-1, mp[1]+1, 0]
        # Calculate kappa
        # kappa = -1/2*(dnxdx1+dnydy1)
        kappa = -1/4*(dnxdx1+dnydy1+dnxdx2+dnydy2)
        print(f'kappa:\n{kappa}')
        # '''

        return [dataset[0], dataset[1], kappa]


class FindAngle(BaseEstimator, TransformerMixin):
    def __init__(self, parameters):
        self.parameters = parameters

    def fit(self, dataset):
        return self
    
    def transform(self, dataset):
        # Seperate dataset
        grad_x = dataset[2].copy()
        grad_y = dataset[3].copy()

        # Get stencil size
        st_sz = self.parameters['stencil_size']

        # Get shape of data
        shape = grad_x.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            grad_x = np.reshape(grad_x, (shape[0], st_sz[0], st_sz[1], 1))
            grad_y = np.reshape(grad_y, (shape[0], st_sz[0], st_sz[1], 1))

        # Calculate angles of negative gradient
        angle_matrix = np.arctan2(-grad_y, -grad_x+1e-10)
        # Transform angle from [-pi, pi] to [0, 1]
        angle_matrix = (angle_matrix + np.pi)*1/(2*np.pi)
        angle_matrix[np.isnan(angle_matrix)] = -1

        if grad_x.shape != shape:
            # Reshape rotated data to original shape
            angle_matrix = np.reshape(angle_matrix, shape)

        return [dataset[0], dataset[1], angle_matrix]


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
        angle_matrix = dataset[2].copy()
        # Get shape of data
        shape = data.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            data = np.reshape(data, (shape[0], st_sz[0], st_sz[1], 1))
            angle_matrix = np.reshape(angle_matrix, (shape[0], st_sz[0], st_sz[1], 1))
        # Calculate midpoint
        mp = [int((st_sz[0]-1)/2), int((st_sz[1]-1)/2)]
        # Get angle in stencil middle
        mid_angle = angle_matrix[:, mp[0], mp[1], 0]
        # Transform angle from [-pi, pi] to [0, 1]
        # mid_angle = (mid_angle + np.pi)*1/(2*np.pi)
        # Calculate rotation based on angle
        rotation = (np.floor(mid_angle*4).astype(int))
        # Get indices where vof values should be rotated 0/90/180/270
        rotation0 = np.argwhere(rotation == 0)
        rotation90 = np.argwhere(rotation == 1)
        rotation180 = np.argwhere(rotation == 2)
        rotation270 = np.argwhere(rotation == 3)
        # Rotate data
        data[rotation0] = data[rotation0]
        data[rotation90] = np.rot90(data[rotation90], 1, axes=(2, 3))
        data[rotation180] = np.rot90(data[rotation180], 2, axes=(2, 3))
        data[rotation270] = np.rot90(data[rotation270], 3, axes=(2, 3))
        # Rotate angle_matrix
        angle_matrix[rotation0] = angle_matrix[rotation0]
        angle_matrix[rotation90] = np.rot90(angle_matrix[rotation90], 1, axes=(2, 3))
        angle_matrix[rotation180] = np.rot90(angle_matrix[rotation180], 2, axes=(2, 3))
        angle_matrix[rotation270] = np.rot90(angle_matrix[rotation270], 3, axes=(2, 3))

        if data.shape != shape:
            # Reshape rotated data to original shape
            data = np.reshape(data, shape)
            angle_matrix = np.reshape(angle_matrix, shape)

        # '''
        # Test
        ind = 6
        # print(f'\ngradient[:,{ind}]: {gradient[:,ind]}')
        print(f'mid_angle[{ind}]: {mid_angle[ind]*180/np.pi}')
        print(f'rotation[{ind}]: {rotation[ind]}')
        # '''
        return [dataset[0], data, dataset[2]]
