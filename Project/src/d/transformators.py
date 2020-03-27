import numpy as np
import time
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
        # dataset = dataset[dataset.iloc[:, 0] > 0]  # Pos values only
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
        time0 = time.time()
        # Get points where gradient should not be calculated
        no_interface = np.nonzero((data == 1) | (data == 0))
        # Initialize gradient matrices
        grad_x = np.empty((data.shape))  # * np.nan
        grad_y = np.empty((data.shape))  # * np.nan
        grad_x.fill(np.nan)
        grad_y.fill(np.nan)
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

    def get_mid_gradient(self, data):
        time0 = time.time()
        # Get stencil size
        st_sz = self.parameters['stencil_size']
        # Calculate midpoint
        mp = [int((st_sz[0]-1)/2), int((st_sz[1]-1)/2)]
        # Calculate gradient in midpoint
        grad_x = np.array([
            data[:, mp[0], mp[1]+1, 0] -
            data[:, mp[0], mp[1]-1, 0]
        ])/2
        grad_y = np.array([
            data[:, mp[0]-1, mp[1], 0] -
            data[:, mp[0]+1, mp[1], 0]
        ])/2

        return [grad_y, grad_x]

    def transform(self, dataset):
        time0 = time.time()
        # Get stencil size
        st_sz = self.parameters['stencil_size']
        # Seperate data from labels
        data = dataset[1].copy()
        # Get shape of data
        shape = data.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            data = np.reshape(data, (shape[0], st_sz[0], st_sz[1], 1))
        if self.parameters['angle']:
            # Get gradient matrix
            [grad_y, grad_x] = self.get_gradient(data)
        else:
            # Get gradient in midpoint only (much faster) if angle matrix is not needed
            [grad_y, grad_x] = self.get_mid_gradient(data)

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
        # Reshape to tensor if angle matrix is needed, otherwise just output vectors
        if (data.shape != shape) & (self.parameters['angle']):
            # Reshape transformed data to original shape
            grad_y = np.reshape(grad_y, shape)
            grad_x = np.reshape(grad_x, shape)

        return [dataset[0], dataset[1], grad_x, grad_y]


class FindAngle(BaseEstimator, TransformerMixin):
    def __init__(self, parameters):
        self.parameters = parameters

    def fit(self, dataset):
        return self
    
    def transform(self, dataset):
        time0 = time.time()
        # Seperate dataset
        grad_x = dataset[2]
        grad_y = dataset[3]

        # Get stencil size
        st_sz = self.parameters['stencil_size']

        # Get shape of data
        shape = grad_x.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if (len(shape) == 2) & (self.parameters['angle']):
            grad_x = np.reshape(grad_x, (shape[0], st_sz[0], st_sz[1], 1))
            grad_y = np.reshape(grad_y, (shape[0], st_sz[0], st_sz[1], 1))

        # Calculate angles of negative gradient (is actually a vector if grad arrays are vectors)
        angle_matrix = np.arctan2(-grad_y, -grad_x+1e-10)
        # Transform angle from [-pi, pi] to [0, 1]
        angle_matrix = (angle_matrix + np.pi)*1/(2*np.pi)
        angle_matrix[np.isnan(angle_matrix)] = -1

        # Reshape to tensor if angle matrix is needed, otherwise just output a vector
        if (grad_x.shape != shape) & (self.parameters['angle']):
            # Reshape rotated data to original shape
            angle_matrix = np.reshape(angle_matrix, shape)

        return [dataset[0], dataset[1], angle_matrix]


class Rotate(BaseEstimator, TransformerMixin):
    def __init__(self, parameters):
        self.parameters = parameters

    def fit(self, dataset):

        return self

    def transform(self, dataset):
        time0 = time.time()
        # Get stencil size
        st_sz = self.parameters['stencil_size']
        # Seperate dataset
        data = dataset[1]
        angle_matrix = dataset[2].copy()
        # Get shape of data
        shape = data.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            data = np.reshape(data, (shape[0], st_sz[0], st_sz[1], 1))
            if self.parameters['angle']:
                angle_matrix = np.reshape(angle_matrix, (shape[0], st_sz[0], st_sz[1], 1))
        if self.parameters['angle']:
            # Calculate midpoint
            mp = [int((st_sz[0]-1)/2), int((st_sz[1]-1)/2)]
            # Get angle in stencil middle
            mid_angle = angle_matrix[:, mp[0], mp[1], 0]
        else:
            # Angle matrix is actually a vector containing all mid_angles if gradients were only calculated in midpoint
            mid_angle = angle_matrix[0]
        # Calculate rotation based on angle
        rotation = (np.floor(mid_angle*4).astype(int))
        # Get indices where vof values should be rotated 0/90/180/270
        # rotation0 = np.argwhere(rotation == 0)
        rotation90 = np.argwhere(rotation == 1)
        rotation180 = np.argwhere(rotation == 2)
        rotation270 = np.argwhere(rotation == 3)
        
        '''
        # Test
        # 3: rot 1
        # 0: rot 2
        # 2: rot 3
        # 5: rot 0
        ind = 2
        print_data_nrt = data[ind].reshape((st_sz[0], st_sz[1])).copy()
        # '''

        # Rotate data
        # data[rotation0] = data[rotation0]
        data[rotation90] = np.rot90(data[rotation90], 1, axes=(3, 2))
        data[rotation180] = np.rot90(data[rotation180], 2, axes=(3, 2))
        data[rotation270] = np.rot90(data[rotation270], 3, axes=(3, 2))
        if self.parameters['angle']:
            # Rotate angle_matrix if it should be included in output
            # angle_matrix[rotation0] = angle_matrix[rotation0]
            angle_matrix[rotation90] = np.rot90(angle_matrix[rotation90], 1, axes=(3, 2))
            angle_matrix[rotation180] = np.rot90(angle_matrix[rotation180], 2, axes=(3, 2))
            angle_matrix[rotation270] = np.rot90(angle_matrix[rotation270], 3, axes=(3, 2))

        if data.shape != shape:
            # Reshape rotated data to original shape
            data = np.reshape(data, shape)
            if self.parameters['angle']:
                angle_matrix = np.reshape(angle_matrix, shape)

        '''
        # Test
        # print(f'\ngradient[:,{ind}]: {gradient[:,ind]}')
        # print(f'mid_angle[{ind}]: {mid_angle[ind]*180/np.pi}')
        print(f'mid_angle[{ind}]:\n{mid_angle[ind]}')
        print(f'rotation[{ind}]: {rotation[ind]}')
        print_data_rot = data[ind].reshape((st_sz[0], st_sz[1])).copy()
        print(f'Data before:\n{print_data_nrt}')
        print(f'Data after:\n{print_data_rot}')
        # '''
        return [dataset[0], data, dataset[2]]


class CDS(BaseEstimator, TransformerMixin):
    def __init__(self, parameters):
        self.parameters = parameters

    def fit(self, dataset):
        return self

    def transform(self, dataset):
        # time0 = time.time()
        # Get stencil size
        st_sz = self.parameters['stencil_size']
        # Seperate data from labels
        data = dataset[1].copy()
        # Get shape of data
        shape = data.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            data = np.reshape(data, (shape[0], st_sz[0], st_sz[1], 1))

        i = int((st_sz[0]-1)/2)
        j = int((st_sz[1]-1)/2)
        c_dx = (data[:, i+1, j, 0] - data[:, i-1, j, 0])/2
        c_dy = (data[:, i, j+1, 0] - data[:, i, j-1, 0])/2

        c_dxx = (data[:, i+2, j, 0] - 2*data[:, i, j, 0] + data[:, i-2, j, 0])/4
        c_dyy = (data[:, i, j+2, 0] - 2*data[:, i, j, 0] + data[:, i, j-2, 0])/4
        c_dxy = (data[:, i+1, j+1, 0] - data[:, i-1, j+1, 0] - data[:, i+1, j-1, 0] + data[:, i-1, j-1, 0])/4

        kappa = -1*np.divide(
            np.multiply(c_dxx, np.multiply(c_dy, c_dy)) -
            np.multiply(c_dxy, np.multiply(c_dx, c_dy))*2 +
            np.multiply(c_dyy, np.multiply(c_dy, c_dy))
            ,
            (np.multiply(c_dx, c_dx) + np.multiply(c_dy, c_dy))**(3/2)
        )

        labels = dataset[0][:10]
        kappa_pr = kappa[:10]

        # print(f'lables:\n{labels}')
        # print(f'kappa_pr:\n{kappa_pr}')

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
        # Reshape to tensor if angle matrix is needed, otherwise just output vectors
        if (data.shape != shape) & (self.parameters['angle']):
            # Reshape transformed data to original shape
            data = np.reshape(data, shape)

        kappa = 1

        return [dataset[0], dataset[1], kappa]


class HF(BaseEstimator, TransformerMixin):
    def __init__(self, parameters):
        self.parameters = parameters

    def fit(self, dataset):
        return self

    def transform(self, dataset):
        # time0 = time.time()
        # Get stencil size
        st_sz = self.parameters['stencil_size']
        # Seperate data from labels
        data = dataset[1].copy()
        grad_x = dataset[2][0]
        grad_y = dataset[3][0]
        # Get shape of data
        shape = data.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            data = np.reshape(data, (shape[0], st_sz[0], st_sz[1], 1))

        # Get midpoint
        i = int((st_sz[0]-1)/2)
        j = int((st_sz[1]-1)/2)

        # Find stencils where gradient points more in y direction (g_y) or x direction (g_x)
        g_y = np.nonzero((np.abs(grad_y) > np.abs(grad_x)) == True)
        g_x = np.nonzero((np.abs(grad_y) > np.abs(grad_x)) == False)

        # Initialize height function vectors
        h1 = np.zeros((data.shape[0]))
        h2 = np.zeros((data.shape[0]))
        h3 = np.zeros((data.shape[0]))
        # Hier läuft noch irgendwas schief
        # Calculate height function values for stencils in y-direction
        for a in np.arange(0, 2*i+1):
            h1[g_y] = np.sum([h1[g_y], data[g_y, a, j-1, 0]], axis=0)
            h2[g_y] = np.sum([h2[g_y], data[g_y, a, j, 0]], axis=0)
            h3[g_y] = np.sum([h3[g_y], data[g_y, a, j+1, 0]], axis=0)
        # Calculate height function values for stencils in x-direction
        for b in np.arange(0, 2*j+1):
            h1[g_x] = np.sum([h1[g_x], data[g_x, i-1, b, 0]], axis=0)
            h2[g_x] = np.sum([h2[g_x], data[g_x, i, b, 0]], axis=0)
            h3[g_x] = np.sum([h3[g_x], data[g_x, i+1, b, 0]], axis=0)

        # Delta = 1/1000  # see data generation
        Delta = 1

        # Calculate derivatives
        h_x = (h1-h3)/(2*Delta)
        h_xx = (h3-2*h2+h1)/(Delta**2)
        # h_x = (h1-h3)/(2)
        # h_xx = (h3-2*h2+h1)

        # Calculate kappa
        kappa = np.round(
            2/Delta*h_xx/((1+np.multiply(h_x, h_x))**(3/2))
            , 3)

        '''
        labels = dataset[0][:10]
        kappa_pr = kappa[:10]
        h_x_pr = h_x[:10]
        h_xx_pr = h_xx[:10]
        h1_pr = h1[:10]
        h2_pr = h2[:10]
        h3_pr = h3[:10]
        g_x_pr = g_x[:10]
        g_y_pr = g_y[:10]

        print(f'h_x_pr:\n{h_x_pr}')
        print(f'h_xx_pr:\n{h_xx_pr}')
        print(f'labels:\n{labels}')
        print(f'kappa_pr:\n{kappa_pr}')
        print(f'h1:\n{h1_pr}')
        print(f'h2:\n{h2_pr}')
        print(f'h3:\n{h3_pr}')
        print(f'g_x:\n{g_x}')
        print(f'g_y:\n{g_y}')
        # '''

        '''
        # Test
        ind = 4

        # print_data_gradx = grad_x.transpose((0, 1, 3, 2))[ind]
        # print_data_grady = grad_y.transpose((0, 1, 3, 2))[ind]
        print_data = data.transpose((0, 1, 3, 2))[ind]
        print_h1 = h1[ind]
        print_h2 = h2[ind]
        print_h3 = h3[ind]
        prt_grad_x = grad_x[ind]
        prt_grad_y = grad_y[ind]
        prt_kappa = kappa[ind]
        # print(f'\nGrad_x:\n{print_data_gradx}')
        # print(f'\nGrad_y:\n{print_data_grady}')
        print(f'\nData:\n{print_data}')
        print(f'h1:\n{print_h1}')
        print(f'h2:\n{print_h2}')
        print(f'h3:\n{print_h3}')
        print(f'prt_grad_x:\n{prt_grad_x}')
        print(f'prt_grad_y:\n{prt_grad_y}')
        print(f'prt_kappa:\n{prt_kappa}')
        # '''
        # Reshape to tensor if angle matrix is needed, otherwise just output vectors
        if (data.shape != shape) & (self.parameters['angle']):
            # Reshape transformed data to original shape
            data = np.reshape(data, shape)

        return [dataset[0], dataset[1], np.array([kappa]).T]


class TwoLayers(BaseEstimator, TransformerMixin):
    def __init__(self, parameters):
        self.parameters = parameters

    def fit(self, dataset):
        return self

    def transform(self, dataset):
        # time0 = time.time()
        # Get stencil size
        st_sz = self.parameters['stencil_size']
        # Seperate data from labels
        data = dataset[1].copy()
        kappa = dataset[2]
        # Get shape of data
        shape = data.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            data = np.reshape(data, (shape[0], st_sz[0], st_sz[1], 1))

        # Get midpoint
        i = int((st_sz[0]-1)/2)
        j = int((st_sz[1]-1)/2)

        kappa_array = np.zeros((data.shape))
        kappa_array[:, i, j, 0] = kappa[:, 0]

        data = np.concatenate((data, kappa_array), axis=3)

        '''
        ind = 0
        print(f'pd_out:\n{output_array[ind, :, :, :]}')
        # '''

        return [dataset[0], data]
