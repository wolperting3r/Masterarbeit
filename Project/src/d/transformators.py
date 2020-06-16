import numpy as np
import time
import sys
from sklearn.base import (
    BaseEstimator,  # for get_params, set_params
    TransformerMixin     # for fit_transform
)
# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)

class TransformData(BaseEstimator, TransformerMixin):
    def __init__(self, parameters, reshape=False):
        self.parameters = parameters
        self.reshape = reshape

    def fit(self, dataset):
        return self  # Nothing to do

    def transform(self, dataset):
        # Split the training and test data into labels (first column) and data
        # dataset = dataset[dataset.iloc[:, 0] > 0]  # Pos values only
        labels = dataset.iloc[:, 0].to_numpy()
        data = dataset.iloc[:, 1:].to_numpy()

        labels = np.round(labels, 5)
        data = np.round(data, 5)

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
        print(f'grad_x.shape:\n{grad_x.shape}')
        print(f'grad_y.shape:\n{grad_y.shape}')
        print(f'data.shape:\n{data.shape}')
        ind = 1
        # print_data_gradx = grad_x.transpose((0, 1, 3, 2))[ind]
        # print_data_grady = grad_y.transpose((0, 1, 3, 2))[ind]
        print_data_gradx = grad_x[:,ind]
        print_data_grady = grad_y[:,ind]
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
    def __init__(self, parameters, rotation_angle=0):  # np.pi/4
        self.parameters = parameters
        self.rotation_angle = rotation_angle

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

        '''
        ind = 6
        print('\nVor der Drehung:')
        print(f'grad_x[:, ind]:\t{grad_x[:, ind]}')
        print(f'grad_y[:, ind]:\t{grad_y[:, ind]}')
        print(f'np.arctan2:\t{np.arctan2(-grad_y, -grad_x+1e-10)[:, ind][0]*180/np.pi}')
        # '''
        if (self.rotation_angle != 0):
            # Rotate gradient vector by rotation_angle
            grad_x_tmp = grad_x.copy()
            grad_y_tmp = grad_y.copy()
            grad_x = np.cos(self.rotation_angle)*grad_x_tmp - np.sin(self.rotation_angle)*grad_y_tmp
            grad_y = np.sin(self.rotation_angle)*grad_x_tmp + np.cos(self.rotation_angle)*grad_y_tmp
            print('Gradient rotiert!')

        '''
        print('\nNach der Drehung:')
        print(f'grad_x[:, ind]:\t{grad_x[:, ind]}')
        print(f'grad_y[:, ind]:\t{grad_y[:, ind]}')
        print(f'np.arctan2:\t{np.arctan2(-grad_y, -grad_x+1e-10)[:, ind][0]*180/np.pi}\n')
        # '''


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
        # rotation = (np.floor(mid_angle*4).astype(int))  # Old
        rotation = (np.floor(mid_angle*8).astype(int))
        # Get indices where vof values should be rotated 0/90/180/270
        # rotation0 = np.argwhere(rotation == 0)
        '''  # Old
        rotation90 = np.argwhere(rotation == 1)
        rotation180 = np.argwhere(rotation == 2)
        rotation270 = np.argwhere(rotation == 3)
        # '''
        # Get indices where vof values should be rotated/flipped along y = -x
        rot0 = np.argwhere(rotation == 7)  # nothing
        rot0f = np.argwhere(rotation == 6)  # flip
        rot90 = np.argwhere(rotation == 5)  # rot 90
        rot90f = np.argwhere(rotation == 4)  # rot 90 + flip
        rot180 = np.argwhere(rotation == 3)  # rot 180
        rot180f = np.argwhere(rotation == 2)  # rot 180 + flip
        rot270 = np.argwhere(rotation == 1)  # rot 270
        rot270f = np.argwhere(rotation == 0)  # rot 270 + flip
        
        '''
        # Test
        # 6: rot 2 3 1
        # 7: rot 3 1 0
        ind = 4
        print_data_nrt = data[ind].reshape((st_sz[0], st_sz[1])).copy()
        # '''

        # Rotate data
        # data[rotation0] = data[rotation0]
        # Rotate by rotation
        '''
        data[rotation90] = np.rot90(data[rotation90], 1, axes=(3, 2))
        data[rotation180] = np.rot90(data[rotation180], 2, axes=(3, 2))
        data[rotation270] = np.rot90(data[rotation270], 3, axes=(3, 2))
        # '''
        # '''
        '''  # Old
        # Rotate by mirroring
        data[rotation90] = np.rot90(data[rotation90], 1, axes=(3, 2))
        data[rotation180] = np.flip(data[rotation180], axis=3)  # flip along y-axis
        data[rotation270] = np.flip(np.rot90(data[rotation270], 1, axes=(3, 2)), axis=3)

        # Flip data along x-axis if sum of first row > sum of last row
        sum_x = np.sum(data, axis=2)
        flip_loc = np.argwhere(sum_x[:, 0, :] > sum_x[:, st_sz[1]-1, :])
        data[flip_loc] = np.flip(data[flip_loc], axis=2)
        # '''
        data[rot0f] = np.transpose(data[rot0f], (0, 1, 3, 2, 4))
        
        data[rot90] = np.rot90(data[rot90], 1, axes=(2, 3))
        data[rot90f] = np.transpose(np.rot90(data[rot90f], 1, axes=(2, 3)), (0, 1, 3, 2, 4))

        data[rot180] = np.rot90(data[rot180], 2, axes=(2, 3))
        data[rot180f] = np.transpose(np.rot90(data[rot180f], 2, axes=(2, 3)), (0, 1, 3, 2, 4))

        data[rot270] = np.rot90(data[rot270], 3, axes=(2, 3))
        data[rot270f] = np.transpose(np.rot90(data[rot270f], 3, axes=(2, 3)), (0, 1, 3, 2, 4))

        # '''
        if self.parameters['angle']:
            # Rotate angle_matrix if it should be included in output
            # angle_matrix[rotation0] = angle_matrix[rotation0]
            '''
            angle_matrix[rotation90] = np.rot90(angle_matrix[rotation90], 1, axes=(3, 2))
            angle_matrix[rotation180] = np.rot90(angle_matrix[rotation180], 2, axes=(3, 2))
            angle_matrix[rotation270] = np.rot90(angle_matrix[rotation270], 3, axes=(3, 2))
            # '''
            # '''
            angle_matrix[rotation90] = np.rot90(angle_matrix[rotation90], 1, axes=(3, 2))
            angle_matrix[rotation180] = np.flip(angle_matrix[rotation180], axis=3)
            angle_matrix[rotation270] = np.flip(np.rot90(angle_matrix[rotation270], 1, axes=(3, 2)), axis=3)
            # '''

        if data.shape != shape:
            # Reshape rotated data to original shape
            data = np.reshape(data, shape)
            if self.parameters['angle']:
                angle_matrix = np.reshape(angle_matrix, shape)

        '''
        # Test
        # print(f'\ngradient[:,{ind}]: {gradient[:,ind]}')
        # print(f'mid_angle[{ind}]: {mid_angle[ind]*180/np.pi}')
        print(f'rotation[{ind}]:\t{rotation[ind]}')
        print_data_rot = data[ind].reshape((st_sz[0], st_sz[1])).copy()
        print(f'Data before:\n{print_data_nrt}')
        print(f'Data after:\n{print_data_rot}')
        # '''
        # return [dataset[0], data, dataset[2]]
        return [dataset[0], data, rotation]


class Shift(BaseEstimator, TransformerMixin):
    def __init__(self, parameters, shift=0, epsilon=0):
        self.parameters = parameters
        self.shift = shift
        self.epsilon = epsilon

    def fit(self, dataset):
        return self

    def transform(self, dataset):
        time0 = time.time()
        # Get stencil size
        st_sz = self.parameters['stencil_size']
        # Seperate dataset
        data = dataset[1]
        # Get shape of data
        shape = data.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            data = np.reshape(data, (shape[0], st_sz[0], st_sz[1], 1))
            if self.parameters['angle']:
                angle_matrix = np.reshape(angle_matrix, (shape[0], st_sz[0], st_sz[1], 1))

        '''
        # Test mit seed 43
        # 1 shift up
        # 4 shift down
        # 21 shift left
        # 9 shift right
        ind = 8
        print_data_nrt = data[ind].reshape((st_sz[0], st_sz[1])).copy()
        print(f'Data before:\n{print_data_nrt}')
        # '''

        for i in range(self.shift):
            # Data can only be shifted where the sum over one row/column is 0/the stencil size (which means there is no information on the interface in that row/column, only 1s or 0s.
            sum_x = np.sum(data, axis=2)
            sum_y = np.sum(data, axis=1)

            # np.random.seed(43)
            # Create array with random integers from 0 to 2. 0 = shift in direction 1, 1 = do not shift, 2 = shift in direction 2
            decider = np.random.randint(0, 3, data.shape[0])

            result = data.copy()
            # Get indices where the data should be shifted
            epsilon = self.epsilon  # war 0.03
            shift_up = np.nonzero((decider == 0) & (sum_x[:, 0, 0] >= (st_sz[1]-epsilon)))
            shift_down = np.nonzero((decider == 2) & (sum_x[:, st_sz[1]-1, 0] <= epsilon))
            shift_right = np.nonzero((decider == 0) & (sum_y[:, st_sz[0]-1, 0] >= (st_sz[0]-epsilon)))
            shift_left = np.nonzero((decider == 2) & (sum_y[:, 0, 0] <= epsilon))

            # Shift the data
            result[shift_up, :st_sz[0]-1, :, :] = data[shift_up, 1:, :, :]
            result[shift_up, st_sz[0]-1, :, :] = 0

            result[shift_down, 1:, :, :] = data[shift_down, :st_sz[0]-1, :, :]
            result[shift_down, 0, :, :] = 1

            result[shift_left, :, :st_sz[0]-1, :] = data[shift_left, :, 1:, :]
            result[shift_left, :, st_sz[0]-1, :] = 1

            result[shift_right, :, 1:, :] = data[shift_right, :, :st_sz[0]-1, :]
            result[shift_right, :, 0, :] = 0

            # Overwrite data
            data = result

        if data.shape != shape:
            # Reshape rotated data to original shape
            data = np.reshape(data, shape)
            if self.parameters['angle']:
                angle_matrix = np.reshape(angle_matrix, shape)

        '''
        # Test
        print_data_rot = result[ind].reshape((st_sz[0], st_sz[1])).copy()
        print(f'Data after:\n{print_data_rot}')
        # '''
        return [dataset[0], data, dataset[2]]


class Edge(BaseEstimator, TransformerMixin):
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

        ''' ENTFERNEN! '''
        data = data[dataset[0] > 0.43]
        print('Teststencil eingefügt')
        teststencil_x = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0.4, 0.601, 0.8, 1, 1, 1, 1, 1, 1],
            [0.2, 0.3, 0.7, 0.9, 1, 1, 1, 0.9, 0.4],
            [0.01, 0.05, 0.1, 0.8, 1, 1, 1, 1, 1],
            [0, 0, 0.4, 0.55, 0.75, 1, 1, 0.6, 0.2],
            [0, 0, 0.4, 0.55, 0.75, 1, 0.9, 0.45, 0.2],
            [0, 0, 0.01, 0.12, 0.45, 0.85, 0.9, 0.45, 0.2],
            [0, 0, 0, 0, 0.1, 0.6, 0.2, 0, 0],
            [0, 0, 0, 0, 0.9, 0.4, 0.9, 0, 0]]
        )
        # teststencil_x = 1-teststencil_x
        # [1, 1, 1, 1, 0.9, 0.6, 0.9, 1, 1],
        teststencil_y = np.rot90(teststencil_x, k=1)
        teststencil_x_rot = np.rot90(teststencil_x, k=2)
        teststencil_y_rot = np.rot90(teststencil_y, k=2)


        teststencil_x = np.reshape(teststencil_x, (1, 81))
        teststencil_y = np.reshape(teststencil_y, (1, 81))
        teststencil_x_rot = np.reshape(teststencil_x_rot, (1, 81))
        teststencil_y_rot = np.reshape(teststencil_y_rot, (1, 81))
        data = np.concatenate((data, teststencil_x, teststencil_y, teststencil_x_rot, teststencil_y_rot), axis=0)
        # print(f'data[0].shape:\t{data[0].shape}')
        # print(f'data[0]:\t{data[0]}')

        # Get shape of data
        shape = data.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            data = np.reshape(data, (shape[0], st_sz[0], st_sz[1], 1))
            if self.parameters['angle']:
                angle_matrix = np.reshape(angle_matrix, (shape[0], st_sz[0], st_sz[1], 1))

        # 1. c < 0.5 -> 0; c > 0.5 -> 1
        # 2. Find 1 value above 0.5, 1 below (e.g. 0.1 0.3 0.6 0.9 -> 0.3, 0.6)
        # 3. Where 

        mask = np.where(data < 0.5, 0, 1)

        mask_x = mask.copy()
        mask_x[:] = np.nan
        mask_y = mask_x.copy()
        # Get closest points to 0.5 coming from above and below (in x- and y-direction)
        for x in range(1, data.shape[1]-1):
            for y in range(1, data.shape[2]-1):
                mask_x[:, y, x, :] = np.logical_or(
                    np.logical_xor(mask[:, y, x+1, :], mask[:, y, x, :]),
                    np.logical_xor(mask[:, y, x-1, :], mask[:, y, x, :])
                )
                mask_y[:, y, x, :] = np.logical_or(
                    np.logical_xor(mask[:, y+1, x, :], mask[:, y, x, :]),
                    np.logical_xor(mask[:, y-1, x, :], mask[:, y, x, :])
                )
        # Cut arrays
        mask_x = mask_x[:, 1:st_sz[0]-1, 1:st_sz[1]-1, :]
        mask_y = mask_y[:, 1:st_sz[0]-1, 1:st_sz[1]-1, :]

        sum_mask_x = np.sum(mask_x, axis=2)
        sum_mask_y = np.sum(mask_y, axis=1)

        and_mask_x = np.zeros((mask_x.shape[0], st_sz[1]-2, st_sz[1]-3, 1))
        and_mask_y = np.zeros((mask_y.shape[0], st_sz[0]-3, st_sz[1]-2, 1))

        # Find adjacent points with sum over and
        for y in range(0, data.shape[1]-3):
            and_mask_y[:, y, :, :] = np.logical_and(mask_y[:, y+1, :, :], mask_y[:, y, :, :])
        for x in range(0, data.shape[1]-3):
            and_mask_x[:, :, x, :] = np.logical_and(mask_x[:, :, x+1, :], mask_x[:, :, x, :])
        sum_and_mask_y = np.sum(and_mask_y, axis=1)  # columns (x) where to write mean on bigger/smaller value
        sum_and_mask_x = np.sum(and_mask_x, axis=2)  # rows (y) where to write mean on bigger/smaller value

        pairs_x = np.empty((0, 2, 3))
        triplets_x = np.empty((0, 3, 3))
        singles_x = np.empty((0, 1, 3))
        valid_combinations = [[1, 0], [2, 0], [2, 1], [3, 1], [3, 2], [4, 2], [4, 3]]

        for combination in valid_combinations:
            # print(f'combination:\t{combination}')
            # Find indices that match the criterion for sum_mask/sum_and_mask alone
            ind_mask_x = np.argwhere((sum_mask_x == combination[0]))
            ind_and_mask_x = np.argwhere((sum_and_mask_x == combination[1]))
            # Find indices that match the combination of both (make rows to single tuples, find common tuples in both lists, extract indices from tuples, sort for first column (stencil number))
            in_both_x = np.array([tpl for tpl in set([tuple(im) for im in ind_mask_x]) & set([tuple(iam) for iam in ind_and_mask_x])])
            # print(f'len(in_both_x):\t{len(in_both_x)}')
            if len(in_both_x) > 0:
                in_both_x = np.array([[ib[0], ib[1], ib[2]] for ib in in_both_x])
                in_both_x = in_both_x[in_both_x[:, 0].argsort()]

                # Oder hier direkt die Indizes finden?
                if tuple(combination) in set([tuple([1, 0]), tuple([2, 0])]):
                    # One single (1 0 0 0 0)
                    # Find columns for given rows where mask_x = 1 (value to be copied)
                    indices = np.argwhere(mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :] == 1)
                    # Overwrite third column of indices with row from in_both_x
                    indices[:, 2] = in_both_x[indices[:, 0]][:, 1]  # rows
                    # Overwrite first column of indices with stencil index from in_both_x
                    indices[:, 0] = in_both_x[indices[:, 0]][:, 0]  # stencils
                    # Append to list
                    '''
                    # Debugging:
                    # print(f'in_both_x[in_both_x[:, 0] == 0]:\n{in_both_x[in_both_x[:, 0] == 0]}')
                    print(f'in_both_x[:10]:\n{in_both_x[:10]}')
                    pdat1 = mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :][:10].reshape((10,5)).copy()
                    print(f'mask_x:\n{pdat1}')

                    print(f'indices:\n{indices[:10]}')
                    print(f'indices:\n{indices[indices[:, 0] == 0]}')
                    # '''
                    # Indices: [stencil nr, column, row]
                    indices = np.reshape(indices, (indices.shape[0], 1, 3))
                    singles_x = np.concatenate((singles_x, indices), axis = 0)

                elif tuple(combination) in set([tuple([2, 1])]):
                    # One pair (0 1 1 0 0)
                    # Caution! First column of indices is index in in_both_x, not in mask_x!
                    # Indices has columns [index in in_both_x, column where value = 1, 0], which makes two rows per one row in in_both_x (because there are two values = 1)
                    indices = np.argwhere(mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :] == 1)
                    # Overwrite third column of indices with row from in_both_x (pair 1)
                    indices[:, 2] = in_both_x[indices[:, 0]][:, 1]  # rows
                    # Overwrite first column of indices with stencil index from in_both_x
                    indices[:, 0] = in_both_x[indices[:, 0]][:, 0]  # stencils
                    # Reshape so pairs are grouped together
                    indices = np.reshape(indices, (int(indices.shape[0]/2), 2, 3))
                    '''
                    # Debugging:
                    # print(f'in_both_x[in_both_x[:, 0] == 0]:\n{in_both_x[in_both_x[:, 0] == 0]}')
                    print(f'in_both_x[:10]:\n{in_both_x[:10]}')
                    pdat1 = mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :][:10].reshape((10,5)).copy()
                    print(f'mask_x:\n{pdat1}')

                    print(f'indices:\n{indices[:10]}')
                    # print(f'indices:\n{indices[indices[:, 0] == 0]}')
                    # '''
                    pairs_x = np.concatenate((pairs_x, indices), axis=0)

                elif tuple(combination) in set([tuple([3, 1])]):
                    # One pair, one single (1 1 0 0 1)
                    # indices = np.argwhere(mask_y[in_both_x[:, 0], in_both_x[:, 1], :, :] == 1)
                    # Get index of first of pair (and_mask = 1)
                    indices_pair_1 = np.argwhere(and_mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :] ==1)
                    # Add second
                    indices_pair_2 = indices_pair_1.copy()
                    indices_pair_2[:, 1] = indices_pair_2[:, 1] + 1

                    # Overwrite third column of indices with row from in_both_x (pair 1)
                    indices_pair_1[:, 2] = in_both_x[indices_pair_1[:, 0]][:, 1]  # rows
                    # Overwrite first column of indices with stencil index from in_both_x
                    indices_pair_1[:, 0] = in_both_x[indices_pair_1[:, 0]][:, 0]  # stencils

                    # Overwrite third column of indices with row from in_both_x (pair 2)
                    indices_pair_2[:, 2] = in_both_x[indices_pair_2[:, 0]][:, 1]  # rows
                    # Overwrite first column of indices with stencil index from in_both_x
                    indices_pair_2[:, 0] = in_both_x[indices_pair_2[:, 0]][:, 0]  # stencils

                    # Stack pairs together
                    indices = np.stack((indices_pair_1, indices_pair_2), axis=1)
                    pairs_x = np.concatenate((pairs_x, indices), axis=0)

                    # '''
                    # Find singles and write into singles
                    indices = np.argwhere((mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :] == 1))

                    # Overwrite third column of indices with row from in_both_x (pair 2)
                    indices[:, 2] = in_both_x[indices[:, 0]][:, 1]  # rows
                    # Overwrite first column of indices with stencil index from in_both_x
                    indices[:, 0] = in_both_x[indices[:, 0]][:, 0]  # stencils

                    # Remove indices that are in pair1 or pair2
                    indices = [x for x in 
                                      set([tuple(x) for x in indices]) - 
                                      set([tuple(x) for x in indices_pair_1]).union(set([tuple(x) for x in indices_pair_2]))
                                     ]
                    indices = np.array([[x[0], x[1], x[2]] for x in np.array(indices)])
                    indices = indices[indices[:, 0].argsort()]

                    indices = np.reshape(indices, (indices.shape[0], 1, 3))
                    singles_x = np.concatenate((singles_x, indices), axis = 0)

                    '''
                    # Debugging:
                    print(f'indices:\n{indices[:10]}')
                    # print(f'in_both_x[in_both_x[:, 0] == 0]:\n{in_both_x[in_both_x[:, 0] == 0]}')
                    print(f'in_both_x[:10]:\n{in_both_x[:10]}')
                    pdat1 = mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :][:10].reshape((10,5)).copy()
                    print(f'mask_x:\n{pdat1}')

                    # print(f'indices:\n{indices[:10]}')
                    # print(f'indices:\n{indices[indices[:, 0] == 0]}')
                    # '''

                elif tuple(combination) in set([tuple([3, 2])]):
                    # One triple (1 1 1 0 0)

                    # Get index of all values
                    indices = np.argwhere(mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :] ==1)
                    # Overwrite third column of indices with row from in_both_x (pair 2)
                    indices[:, 2] = in_both_x[indices[:, 0]][:, 1]  # rows
                    # Overwrite first column of indices with stencil index from in_both_x
                    indices[:, 0] = in_both_x[indices[:, 0]][:, 0]  # stencils
                    # Reshape so triplets are grouped together
                    indices = np.reshape(indices, (int(indices.shape[0]/3), 3, 3))

                    '''
                    # Debugging:
                    print(f'indices:\n{indices[:10]}')
                    # print(f'in_both_x[in_both_x[:, 0] == 0]:\n{in_both_x[in_both_x[:, 0] == 0]}')
                    print(f'in_both_x[:10]:\n{in_both_x[:10]}')
                    pdat1 = mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :][:10].reshape((10,5)).copy()
                    print(f'mask_x:\n{pdat1}')
                    # '''
                    triplets_x = np.concatenate((triplets_x, indices), axis=0)

                elif tuple(combination) in set([tuple([4, 2])]):
                    # Two pairs, seperated (1 1 0 1 1)
                    # Get index of first of pair (and_mask = 1)
                    indices_pair_1 = np.argwhere(and_mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :] ==1)
                    # Add second
                    indices_pair_2 = indices_pair_1.copy()
                    indices_pair_2[:, 1] = indices_pair_2[:, 1] + 1

                    # Overwrite third column of indices with row from in_both_x (pair 1)
                    indices_pair_1[:, 2] = in_both_x[indices_pair_1[:, 0]][:, 1]  # rows
                    # Overwrite first column of indices with stencil index from in_both_x
                    indices_pair_1[:, 0] = in_both_x[indices_pair_1[:, 0]][:, 0]  # stencils

                    # Overwrite third column of indices with row from in_both_x (pair 2)
                    indices_pair_2[:, 2] = in_both_x[indices_pair_2[:, 0]][:, 1]  # rows
                    # Overwrite first column of indices with stencil index from in_both_x
                    indices_pair_2[:, 0] = in_both_x[indices_pair_2[:, 0]][:, 0]  # stencils

                    # Stack pairs together
                    indices = np.stack((indices_pair_1, indices_pair_2), axis=1)

                    '''
                    # Debugging:
                    print(f'indices[:10]:\t{indices[:10]}')
                    # print(f'in_both_x[in_both_x[:, 0] == 0]:\n{in_both_x[in_both_x[:, 0] == 0]}')
                    print(f'in_both_x[:10]:\n{in_both_x[:10]}')
                    pdat1 = mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :][:10].reshape((10,5)).copy()
                    print(f'mask_x:\n{pdat1}')
                    # '''
                    pairs_x = np.concatenate((pairs_x, indices), axis=0)

                elif tuple(combination) in set([tuple([4, 3])]):
                    # Two pairs, adjacent (1 1 1 1 0)

                    # Get index of first of pair (and_mask = 1)
                    indices_pair_1 = np.argwhere(and_mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :] ==1)

                    # Overwrite third column of indices with row from in_both_x (pair 1)
                    indices_pair_1[:, 2] = in_both_x[indices_pair_1[:, 0]][:, 1]  # rows
                    # Overwrite first column of indices with stencil index from in_both_x
                    indices_pair_1[:, 0] = in_both_x[indices_pair_1[:, 0]][:, 0]  # stencils
                    # Get first three
                    indices_pair_1 = np.reshape(indices_pair_1, (int(indices_pair_1.shape[0]/3), 3, 3))
                    # Remove middle one to get groups of two
                    indices_pair_1 = np.delete(indices_pair_1, 1, 1)
                    # Break up groups of two (both being pair 1 of one of the pairs)
                    indices_pair_1 = np.reshape(indices_pair_1, (int(indices_pair_1.shape[0]*2), 3))
                    # Generate pair 2 corresponding to pair 1
                    indices_pair_2 = indices_pair_1.copy()
                    indices_pair_2[:, 1] = indices_pair_2[:, 1] + 1
                    # Stack pairs together
                    indices = np.stack((indices_pair_1, indices_pair_2), axis=1)

                    '''
                    print(f'indices_pair_1[:10]:\n{indices_pair_1[:10]}')
                    print(f'indices[:10]:\n{indices[:10]}')
                    # Debugging:
                    # print(f'indices[:10]:\t{indices[:10]}')
                    print(f'in_both_x[:10]:\n{in_both_x[:10]}')
                    pdat1 = mask_x[in_both_x[:, 0], in_both_x[:, 1], :, :][:10].reshape((10,5)).copy()
                    print(f'mask_x:\n{pdat1}')
                    # '''
                    pairs_x = np.concatenate((pairs_x, indices), axis=0)


        pairs_y = np.empty((0, 2, 3))
        triplets_y = np.empty((0, 3, 3))
        singles_y = np.empty((0, 1, 3))
        # print('\ny\n')
        for combination in valid_combinations:
            # print(f'combination:\t{combination}')
            # Do the same for y
            ind_mask_y = np.argwhere((sum_mask_y == combination[0]))
            ind_and_mask_y = np.argwhere((sum_and_mask_y == combination[1]))
            in_both_y = [tpl for tpl in set([tuple(im) for im in ind_mask_y]) & set([tuple(iam) for iam in ind_and_mask_y])]
            in_both_y = np.array([[ib[0], ib[1], ib[2]] for ib in np.array(in_both_y)])
            in_both_y = in_both_y[in_both_y[:, 0].argsort()]

            # print(f'len(in_both_y):\t{len(in_both_y)}')
            if len(in_both_y) > 0:
                in_both_y = np.array([[ib[0], ib[1], ib[2]] for ib in in_both_y])
                in_both_y = in_both_y[in_both_y[:, 0].argsort()]

                # Oder hier direkt die Indizes finden?
                if tuple(combination) in set([tuple([1, 0]), tuple([2, 0])]):
                    # One single (1 0 0 0 0)
                    # Find columns for given rows where mask_x = 1 (value to be copied)
                    indices = np.argwhere(mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :] == 1)
                    # Overwrite third column of indices with row from in_both_y
                    indices[:, 2] = indices[:, 1]  # rows
                    indices[:, 1] = in_both_y[indices[:, 0]][:, 1]  # columns
                    # Overwrite first column of indices with stencil index from in_both_y
                    indices[:, 0] = in_both_y[indices[:, 0]][:, 0]  # stencils
                    # Append to list
                    '''
                    # Debugging:
                    # print(f'in_both_y[in_both_y[:, 0] == 0]:\n{in_both_y[in_both_y[:, 0] == 0]}')
                    print(f'in_both_y[:10]:\n{in_both_y[:10]}')
                    pdat1 = mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :][:10].reshape((10,5)).copy()
                    print(f'mask_y:\n{pdat1}')

                    print(f'indices:\n{indices[:10]}')
                    print(f'indices:\n{indices[indices[:, 0] == 0]}')
                    # '''
                    # Indices: [stencil nr, column, row]
                    indices = np.reshape(indices, (indices.shape[0], 1, 3))
                    singles_y = np.concatenate((singles_y, indices), axis = 0)

                elif tuple(combination) in set([tuple([2, 1])]):
                    # One pair (0 1 1 0 0)
                    # Caution! First column of indices is index in in_both_y, not in mask_x!
                    # Indices has columns [index in in_both_y, column where value = 1, 0], which makes two rows per one row in in_both_y (because there are two values = 1)
                    indices = np.argwhere(mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :] == 1)
                    # Overwrite third column of indices with row from in_both_y (pair 1)
                    indices[:, 2] = indices[:, 1]  # rows
                    indices[:, 1] = in_both_y[indices[:, 0]][:, 1]  # columns
                    # Overwrite first column of indices with stencil index from in_both_y
                    indices[:, 0] = in_both_y[indices[:, 0]][:, 0]  # stencils
                    # Reshape so pairs are grouped together
                    indices = np.reshape(indices, (int(indices.shape[0]/2), 2, 3))
                    '''
                    # Debugging:
                    # print(f'in_both_y[in_both_y[:, 0] == 0]:\n{in_both_y[in_both_y[:, 0] == 0]}')
                    print(f'in_both_y[:10]:\n{in_both_y[:10]}')
                    pdat1 = mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :][:10].reshape((10,5)).copy()
                    print(f'mask_y:\n{pdat1}')

                    print(f'indices:\n{indices[:10]}')
                    # print(f'indices:\n{indices[indices[:, 0] == 0]}')
                    # '''
                    pairs_y = np.concatenate((pairs_y, indices), axis=0)

                elif tuple(combination) in set([tuple([3, 1])]):
                    ''' Scheint es bei x nicht zu geben, bei y testen '''
                    # One pair, one single (1 1 0 0 1)
                    # indices = np.argwhere(mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :] == 1)
                    # Get index of first of pair (and_mask = 1)
                    indices_pair_1 = np.argwhere(and_mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :] ==1)
                    # Add second
                    indices_pair_2 = indices_pair_1.copy()
                    indices_pair_2[:, 1] = indices_pair_2[:, 1] + 1

                    # Overwrite third column of indices with row from in_both_y (pair 1)
                    indices_pair_1[:, 2] = indices_pair_1[:, 1]  # rows
                    indices_pair_1[:, 1] = in_both_y[indices_pair_1[:, 0]][:, 1]  # columns
                    # Overwrite first column of indices with stencil index from in_both_y
                    indices_pair_1[:, 0] = in_both_y[indices_pair_1[:, 0]][:, 0]  # stencils

                    # Overwrite third column of indices with row from in_both_y (pair 2)
                    indices_pair_2[:, 2] = indices_pair_2[:, 1]  # rows
                    indices_pair_2[:, 1] = in_both_y[indices_pair_2[:, 0]][:, 1]  # columns
                    # Overwrite first column of indices with stencil index from in_both_y
                    indices_pair_2[:, 0] = in_both_y[indices_pair_2[:, 0]][:, 0]  # stencils

                    # Stack pairs together
                    indices = np.stack((indices_pair_1, indices_pair_2), axis=1)
                    pairs_y = np.concatenate((pairs_y, indices), axis=0)

                    # '''
                    # Find singles and write into singles
                    indices = np.argwhere((mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :] == 1))

                    # Overwrite third column of indices with row from in_both_y (pair 2)
                    indices[:, 2] = indices[:, 1]  # rows
                    indices[:, 1] = in_both_y[indices[:, 0]][:, 1]  # columns
                    # Overwrite first column of indices with stencil index from in_both_y
                    indices[:, 0] = in_both_y[indices[:, 0]][:, 0]  # stencils

                    # Remove indices that are in pair1 or pair2
                    indices = [x for x in 
                                      set([tuple(x) for x in indices]) - 
                                      set([tuple(x) for x in indices_pair_1]).union(set([tuple(x) for x in indices_pair_2]))
                                     ]
                    indices = np.array([[x[0], x[1], x[2]] for x in np.array(indices)])
                    indices = indices[indices[:, 0].argsort()]

                    indices = np.reshape(indices, (indices.shape[0], 1, 3))
                    singles_y = np.concatenate((singles_y, indices), axis = 0)

                    '''
                    # Debugging:
                    print(f'indices_single:\n{indices_single[:10]}')
                    # print(f'indices_pair_1:\n{indices_pair_1[:10]}')
                    # print(f'indices_pair_2:\n{indices_pair_2[:10]}')
                    # print(f'in_both_y[in_both_y[:, 0] == 0]:\n{in_both_y[in_both_y[:, 0] == 0]}')
                    print(f'in_both_y[:10]:\n{in_both_y[:10]}')
                    pdat1 = mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :][:10].reshape((10,5)).copy()
                    print(f'mask_y:\n{pdat1}')

                    # print(f'indices:\n{indices[:10]}')
                    # print(f'indices:\n{indices[indices[:, 0] == 0]}')
                    # '''

                elif tuple(combination) in set([tuple([3, 2])]):
                    # One triple (1 1 1 0 0)

                    # Get index of all values
                    indices = np.argwhere(mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :] ==1)
                    # Overwrite third column of indices with row from in_both_y (pair 2)
                    indices[:, 2] = indices[:, 1]  # rows
                    indices[:, 1] = in_both_y[indices[:, 0]][:, 1]  # columns
                    # Overwrite first column of indices with stencil index from in_both_y
                    indices[:, 0] = in_both_y[indices[:, 0]][:, 0]  # stencils
                    # Reshape so triplets are grouped together
                    indices = np.reshape(indices, (int(indices.shape[0]/3), 3, 3))

                    '''
                    # Debugging:
                    print(f'indices:\n{indices[:10]}')
                    # print(f'in_both_y[in_both_y[:, 0] == 0]:\n{in_both_y[in_both_y[:, 0] == 0]}')
                    print(f'in_both_y[:10]:\n{in_both_y[:10]}')
                    pdat1 = mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :][:10].reshape((10,5)).copy()
                    print(f'mask_y:\n{pdat1}')
                    # '''
                    triplets_y = np.concatenate((triplets_y, indices), axis=0)

                elif tuple(combination) in set([tuple([4, 2])]):
                    # Two pairs, seperated (1 1 0 1 1)
                    # Get index of first of pair (and_mask = 1)
                    indices_pair_1 = np.argwhere(and_mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :] ==1)
                    # Add second
                    indices_pair_2 = indices_pair_1.copy()
                    indices_pair_2[:, 1] = indices_pair_2[:, 1] + 1

                    # Overwrite third column of indices with row from in_both_y (pair 1)
                    indices_pair_1[:, 2] = indices_pair_1[:, 1]  # rows
                    indices_pair_1[:, 1] = in_both_y[indices_pair_1[:, 0]][:, 1]  # columns
                    # Overwrite first column of indices with stencil index from in_both_y
                    indices_pair_1[:, 0] = in_both_y[indices_pair_1[:, 0]][:, 0]  # stencils

                    # Overwrite third column of indices with row from in_both_y (pair 2)
                    indices_pair_2[:, 2] = indices_pair_2[:, 1]  # rows
                    indices_pair_2[:, 1] = in_both_y[indices_pair_2[:, 0]][:, 1]  # columns
                    # Overwrite first column of indices with stencil index from in_both_y
                    indices_pair_2[:, 0] = in_both_y[indices_pair_2[:, 0]][:, 0]  # stencils

                    # Stack pairs together
                    indices = np.stack((indices_pair_1, indices_pair_2), axis=1)

                    '''
                    # Debugging:
                    print(f'indices[:10]:\t{indices[:10]}')
                    # print(f'in_both_y[in_both_y[:, 0] == 0]:\n{in_both_y[in_both_y[:, 0] == 0]}')
                    print(f'in_both_y[:10]:\n{in_both_y[:10]}')
                    pdat1 = mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :][:10].reshape((10,5)).copy()
                    print(f'mask_y:\n{pdat1}')
                    # '''
                    pairs_y = np.concatenate((pairs_y, indices), axis=0)

                elif tuple(combination) in set([tuple([4, 3])]):
                    # Two pairs, adjacent (1 1 1 1 0)

                    # Get index of first of pair (and_mask = 1)
                    indices_pair_1 = np.argwhere(and_mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :] ==1)

                    # Overwrite third column of indices with row from in_both_y (pair 1)
                    indices_pair_1[:, 2] = indices_pair_1[:, 1]  # rows
                    indices_pair_1[:, 1] = in_both_y[indices_pair_1[:, 0]][:, 1]  # columns
                    # Overwrite first column of indices with stencil index from in_both_y
                    indices_pair_1[:, 0] = in_both_y[indices_pair_1[:, 0]][:, 0]  # stencils
                    # Get first three
                    indices_pair_1 = np.reshape(indices_pair_1, (int(indices_pair_1.shape[0]/3), 3, 3))
                    # Remove middle one to get groups of two
                    indices_pair_1 = np.delete(indices_pair_1, 1, 1)
                    # Break up groups of two (both being pair 1 of one of the pairs)
                    indices_pair_1 = np.reshape(indices_pair_1, (int(indices_pair_1.shape[0]*2), 3))
                    # Generate pair 2 corresponding to pair 1
                    indices_pair_2 = indices_pair_1.copy()
                    indices_pair_2[:, 2] = indices_pair_2[:, 2] + 1
                    # Stack pairs together
                    indices = np.stack((indices_pair_1, indices_pair_2), axis=1)

                    '''
                    print(f'indices_pair_1[:10]:\n{indices_pair_1[:10]}')
                    print(f'indices[:10]:\n{indices[:10]}')
                    # Debugging:
                    # print(f'indices[:10]:\t{indices[:10]}')
                    # print(f'in_both_y[in_both_y[:, 0] == 0]:\n{in_both_y[in_both_y[:, 0] == 0]}')
                    print(f'in_both_y[:10]:\n{in_both_y[:10]}')
                    pdat1 = mask_y[in_both_y[:, 0], :, in_both_y[:, 1], :][:10].reshape((10,5)).copy()
                    print(f'mask_y:\n{pdat1}')
                    # '''
                    pairs_y = np.concatenate((pairs_y, indices), axis=0)

        # Make rowcol arrays integer
        singles_x = singles_x.astype(np.int)
        pairs_x = pairs_x.astype(np.int)
        triplets_x = triplets_x.astype(np.int)
        singles_y = singles_y.astype(np.int)
        pairs_y = pairs_y.astype(np.int)
        triplets_y = triplets_y.astype(np.int)

        # a = b

        # Find indices for each case, append it to either a list of pairs (mean) or a list of single values
        # 
        # Go through cases, append index pairs
        # If second index = 0 -10 -10, just take the value, otherwise take the mean
        # mean -> on bigger/on smaller

        # 4:    0 1 1 1 1 1 1 (sm = 1)          -> übernehmen (4818)
        #         1 0 0 0 0

        # 2:    0 0 0 0 1 1 1 (sm = 2, sam = 1) -> interpolieren (13276)
        #         0 0 1 1 0 

        # 34:   0 1 1 1 1 1 0 (sm = 2, sam = 0) -> übernehmen (242)
        #         1 0 0 0 1

        # 0:    0 0 1 1 1 1 0 (sm = 3, sam = 1) -> interpolieren & übernehmen (645)
        #         1 1 0 0 1

        # 76:   0 0 0 0 1 0 0 (sm = 3, sam = 2) -> 2x interpolieren, Mittelwert auf Mitte/beide Ränder (297)
        #         0 0 1 1 1

        # 35:   0 0 1 1 1 0 0 (sm = 4, sam = 2) -> in zwei aufteilen, interpolieren (178)
        #         1 1 0 1 1

        # 51:   0 0 0 1 1 0 0 (sm = 4, sam = 3) -> in zwei aufteilen, interpolieren (189)
        #         0 1 1 1 1 

        '''
        ind = 51
        print(f'\nINDEX = {ind}\n')
        pdat1 = data[ind].reshape((st_sz[0], st_sz[1])).copy()
        print(f'pdat1:\n{pdat1}')
        pdat2 = mask[ind].reshape((st_sz[0], st_sz[1])).copy()
        print(f'pdat2:\n{pdat2}')
        pdat2 = mask_x[:, :, :, :][ind].reshape((st_sz[0]-2, st_sz[0]-2)).copy()
        print(f'mask_x:\n{pdat2}')
        pdat2 = mask_y[:, :, :, :][ind].reshape((st_sz[0]-2, st_sz[0]-2)).copy()
        print(f'mask_y:\n{pdat2}')
        pdat2 = sum_mask_x[ind].reshape((st_sz[0]-2)).copy()
        print(f'sum_mask_x:\n{pdat2}')
        pdat2 = sum_mask_y[ind].reshape((st_sz[0]-2)).copy()
        print(f'sum_mask_y:\n{pdat2}')
        pdat2 = and_mask_x[:, :, :, :][ind].reshape((st_sz[0]-2, st_sz[0]-3)).copy()
        print(f'and_mask_x:\n{pdat2}')
        pdat2 = and_mask_y[:, :, :, :][ind].reshape((st_sz[0]-3, st_sz[0]-2)).copy()
        print(f'and_mask_y:\n{pdat2}')
        pdat2 = sum_and_mask_x[ind].reshape((st_sz[0]-2)).copy()
        print(f'sum_and_mask_x:\n{pdat2}')
        pdat2 = sum_and_mask_y[ind].reshape((st_sz[0]-2)).copy()
        print(f'sum_and_mask_y:\n{pdat2}')

        print(f'np.unique(sum_mask_x):\t{np.unique(sum_mask_x)}')
        print(f'np.unique(sum_mask_y):\t{np.unique(sum_mask_y)}')
        print(f'np.unique(sum_and_mask_x):\t{np.unique(sum_and_mask_x)}')
        print(f'np.unique(sum_and_mask_y):\t{np.unique(sum_and_mask_y)}')
        print(f'\nINDEX = {ind}\n')
        print(f'pairs_x:\n{pairs_x}')
        print(f'pairs_x.shape:\t{pairs_x.shape}')
        print(f'triplets_x.shape:\t{triplets_x.shape}')
        print(f'singles_x.shape:\t{singles_x.shape}')
        print(f'pairs_y:\n{pairs_y}')
        print(f'pairs_y.shape:\t{pairs_y.shape}')
        print(f'triplets_y.shape:\t{triplets_y.shape}')
        print(f'singles_y.shape:\t{singles_y.shape}')
        # '''

        # Get vof values of points closest to 0.5
        # data_x = np.multiply(mask_x, data[:, 1:st_sz[0]-1, 1:st_sz[1]-1, :])
        # data_y = np.multiply(mask_y, data[:, 1:st_sz[0]-1, 1:st_sz[1]-1, :])

        # Get inner stencil from data
        data_cut = data[:, 1:st_sz[0]-1, 1:st_sz[1]-1, :]

        # Initialize output stencils
        interp_x = data_cut.copy()
        interp_x[:] = 0
        interp_y = data_cut.copy()
        interp_y[:] = 0

        # PAIRS

        # Get data of both parts of pairs
        pairs1_x = data_cut[pairs_x[:, 0, 0], pairs_x[:, 0, 2], pairs_x[:, 0, 1], 0]
        pairs2_x = data_cut[pairs_x[:, 1, 0], pairs_x[:, 1, 2], pairs_x[:, 1, 1], 0]

        pairs1_y = data_cut[pairs_y[:, 0, 0], pairs_y[:, 0, 2], pairs_y[:, 0, 1], 0]
        pairs2_y = data_cut[pairs_y[:, 1, 0], pairs_y[:, 1, 2], pairs_y[:, 1, 1], 0]

        # Calculate mean of pairs
        mean_x = (pairs1_x + pairs2_x)/2
        mean_y = (pairs1_y + pairs2_y)/2

        # Decide weather mean should be written on bigger or smaller value
        on_bigger_x = np.where(mean_x < 0.5, mean_x + 0.5, 0)
        on_smaller_x = np.where(mean_x >= 0.5, mean_x - 0.5, 0)
        on_bigger_y = np.where(mean_y < 0.5, mean_y + 0.5, 0)
        on_smaller_y = np.where(mean_y >= 0.5, mean_y - 0.5, 0)

        # Find out where p1 > p2 and where p1 <= p2
        p1_bigger_x = np.where(pairs1_x > pairs2_x, 1, 0)
        p1_smaller_x = np.where(pairs1_x <= pairs2_x, 1, 0)
        p1_bigger_y = np.where(pairs1_y > pairs2_y, 1, 0)
        p1_smaller_y = np.where(pairs1_y <= pairs2_y, 1, 0)

        # Get values that should be written on p1/p2
        int_p1_x = np.multiply(p1_bigger_x, on_bigger_x) + np.multiply(p1_smaller_x, on_smaller_x)
        int_p2_x = np.multiply(p1_bigger_x, on_smaller_x) + np.multiply(p1_smaller_x, on_bigger_x)
        int_p1_y = np.multiply(p1_bigger_y, on_bigger_y) + np.multiply(p1_smaller_y, on_smaller_y)
        int_p2_y = np.multiply(p1_bigger_y, on_smaller_y) + np.multiply(p1_smaller_y, on_bigger_y)

        interp_x[pairs_x[:, 0, 0], pairs_x[:, 0, 2], pairs_x[:, 0, 1], 0] = int_p1_x
        interp_x[pairs_x[:, 1, 0], pairs_x[:, 1, 2], pairs_x[:, 1, 1], 0] = int_p2_x
        interp_y[pairs_y[:, 0, 0], pairs_y[:, 0, 2], pairs_y[:, 0, 1], 0] = int_p1_y
        interp_y[pairs_y[:, 1, 0], pairs_y[:, 1, 2], pairs_y[:, 1, 1], 0] = int_p2_y

        '''
        # Debug output
        tmp = np.stack((pairs1_x, pairs2_x, mean_x, on_bigger_x, on_smaller_x, p1_bigger_x, p1_smaller_x, int_p1_x, int_p2_x, pairs_x[:, 0, 0], pairs_x[:, 0, 1], pairs_x[:, 0, 2], pairs_x[:, 1, 0], pairs_x[:, 1, 1], pairs_x[:, 1, 2]), axis=1)
        print('pairs1_x, pairs2_x, mean_x, on_bigger_x, on_smaller_x, p1_bigger_x, p1_smaller_x, int_p1_x, int_p2_x, pairs_x[:, 0, :], pairs_x[:, 1, :]')
        with np.printoptions(threshold=np.inf):
            print(f'{np.round(tmp[:20],3)}')
        tmp = np.stack((pairs1_y, pairs2_y, mean_y, on_bigger_y, on_smaller_y, p1_bigger_y, p1_smaller_y, int_p1_y, int_p2_y, pairs_y[:, 0, 0], pairs_y[:, 0, 1], pairs_y[:, 0, 2], pairs_y[:, 1, 0], pairs_y[:, 1, 1], pairs_y[:, 1, 2]), axis=1)
        print('pairs1_y, pairs2_y, mean_y, on_bigger_y, on_smaller_y, p1_bigger_y, p1_smaller_y, int_p1_y, int_p2_y, pairs_y[:, 0, :], pairs_y[:, 1, :]')
        with np.printoptions(threshold=np.inf):
            print(f'{np.round(tmp[:20],3)}')
        # print(f'value_on_bigger[:10]:\n{value_on_bigger[:10]}')
        print(f'tmp[tmp[:, 2] == 0]:\t{tmp[tmp[:, 2] == 0]}')  # should be empty
        print(f'tmp[tmp[:, 1] == 0]:\t{tmp[tmp[:, 1] == 0]}')  # should be empty
        print(f'tmp[tmp[:, 0] == 0]:\t{tmp[tmp[:, 0] == 0]}')  # should be empty
        # '''

        # TRIPLETS

        # Get data of both parts of pairs
        triplets1_x = data_cut[triplets_x[:, 0, 0], triplets_x[:, 0, 2], triplets_x[:, 0, 1], 0]
        triplets2_x = data_cut[triplets_x[:, 1, 0], triplets_x[:, 1, 2], triplets_x[:, 1, 1], 0]
        triplets3_x = data_cut[triplets_x[:, 2, 0], triplets_x[:, 2, 2], triplets_x[:, 2, 1], 0]

        triplets1_y = data_cut[triplets_y[:, 0, 0], triplets_y[:, 0, 2], triplets_y[:, 0, 1], 0]
        triplets2_y = data_cut[triplets_y[:, 1, 0], triplets_y[:, 1, 2], triplets_y[:, 1, 1], 0]
        triplets3_y = data_cut[triplets_y[:, 2, 0], triplets_y[:, 2, 2], triplets_y[:, 2, 1], 0]

        # Calculate mean of pairs
        mean_x_1 = (triplets1_x + triplets2_x)/2
        mean_x_2 = (triplets2_x + triplets3_x)/2

        mean_y_1 = (triplets1_y + triplets2_y)/2
        mean_y_2 = (triplets2_y + triplets3_y)/2

        # Decide weather mean should be written on bigger or smaller value
        on_bigger_x_1 = np.where(mean_x_1 < 0.5, mean_x_1 + 0.5, 0)
        on_bigger_x_2 = np.where(mean_x_2 < 0.5, mean_x_2 + 0.5, 0)
        on_smaller_x_1 = np.where(mean_x_1 >= 0.5, mean_x_1 - 0.5, 0)
        on_smaller_x_2 = np.where(mean_x_2 >= 0.5, mean_x_2 - 0.5, 0)

        on_bigger_y_1 = np.where(mean_y_1 < 0.5, mean_y_1 + 0.5, 0)
        on_bigger_y_2 = np.where(mean_y_2 < 0.5, mean_y_2 + 0.5, 0)
        on_smaller_y_1 = np.where(mean_y_1 >= 0.5, mean_y_1 - 0.5, 0)
        on_smaller_y_2 = np.where(mean_y_2 >= 0.5, mean_y_2 - 0.5, 0)

        # Find out where p1 > p2 and where p1 <= p2
        t1_bigger_x_1 = np.where(triplets1_x > triplets2_x, 1, 0)
        t1_smaller_x_1 = np.where(triplets1_x <= triplets2_x, 1, 0)
        t2_bigger_x_2 = np.where(triplets2_x > triplets3_x, 1, 0)
        t2_smaller_x_2 = np.where(triplets2_x <= triplets3_x, 1, 0)

        t1_bigger_y_1 = np.where(triplets1_y > triplets2_y, 1, 0)
        t1_smaller_y_1 = np.where(triplets1_y <= triplets2_y, 1, 0)
        t2_bigger_y_2 = np.where(triplets2_y > triplets3_y, 1, 0)
        t2_smaller_y_2 = np.where(triplets2_y <= triplets3_y, 1, 0)

        # Get values that should be written on p1/p2
        int_t1_x = np.multiply(t1_bigger_x_1, on_bigger_x_1) + np.multiply(t1_smaller_x_1, on_smaller_x_1)
        int_t3_x = np.multiply(t2_bigger_x_2, on_smaller_x_2) + np.multiply(t2_smaller_x_2, on_bigger_x_2)

        int_t1_y = np.multiply(t1_bigger_y_1, on_bigger_y_1) + np.multiply(t1_smaller_y_1, on_smaller_y_1)
        int_t3_y = np.multiply(t2_bigger_y_2, on_smaller_y_2) + np.multiply(t2_smaller_y_2, on_bigger_y_2)

        # if middle value of triplet should be overwritten with left and right mean, overwrite it with the mean of both means
        int_t2_x_1 = np.multiply(t1_bigger_x_1, on_smaller_x_1) + np.multiply(t1_smaller_x_1, on_bigger_x_1)
        int_t2_x_2 = np.multiply(t2_bigger_x_2, on_bigger_x_2) + np.multiply(t2_smaller_x_2, on_smaller_x_2)
        int_t2_x = np.where(((int_t2_x_1 != 0) & (int_t2_x_2 != 0)), (int_t2_x_1+int_t2_x_2)/2, int_t2_x_1+int_t2_x_2)

        int_t2_y_1 = np.multiply(t1_bigger_y_1, on_smaller_y_1) + np.multiply(t1_smaller_y_1, on_bigger_y_1)
        int_t2_y_2 = np.multiply(t2_bigger_y_2, on_bigger_y_2) + np.multiply(t2_smaller_y_2, on_smaller_y_2)
        int_t2_y = np.where(((int_t2_y_1 != 0) & (int_t2_y_2 != 0)), (int_t2_y_1+int_t2_y_2)/2, int_t2_y_1+int_t2_y_2)

        # Write values to output stencils
        interp_x[triplets_x[:, 0, 0], triplets_x[:, 0, 2], triplets_x[:, 0, 1], 0] = int_t1_x
        interp_x[triplets_x[:, 1, 0], triplets_x[:, 1, 2], triplets_x[:, 1, 1], 0] = int_t2_x
        interp_x[triplets_x[:, 2, 0], triplets_x[:, 2, 2], triplets_x[:, 2, 1], 0] = int_t3_x

        interp_y[triplets_y[:, 0, 0], triplets_y[:, 0, 2], triplets_y[:, 0, 1], 0] = int_t1_y
        interp_y[triplets_y[:, 1, 0], triplets_y[:, 1, 2], triplets_y[:, 1, 1], 0] = int_t2_y
        interp_y[triplets_y[:, 2, 0], triplets_y[:, 2, 2], triplets_y[:, 2, 1], 0] = int_t3_y

        '''
        tmp = np.stack((triplets1_x, triplets2_x, triplets3_x, mean_x_1, mean_x_2, on_bigger_x_1, on_bigger_x_2, on_smaller_x_1, on_smaller_x_2, t1_bigger_x_1, t1_smaller_x_1, t2_bigger_x_2, t2_smaller_x_2, int_t1_x, int_t2_x_1, int_t2_x, int_t2_x_2, int_t3_x, ), axis=1)
        print('triplets1_x, triplets2_x, triplets3_x, mean_x_1, mean_x_2, on_bigger_x_1, on_bigger_x_2, on_smaller_x_1, on_smaller_x_2, t1_bigger_x_1, t1_smaller_x_1, t2_bigger_x_2, t2_smaller_x_2, int_t1_x, int_t2_x_1, int_t2_x, int_t2_x_2, int_t3_x, ')
        with np.printoptions(threshold=np.inf):
            print(f'{np.round(tmp[:20],3)}')
        # '''

        # Singles
        # Just copy single values from data to interp
        interp_x[singles_x[:, 0, 0], singles_x[:, 0, 2], singles_x[:, 0, 1], 0] = data_cut[singles_x[:, 0, 0], singles_x[:, 0, 2], singles_x[:, 0, 1], 0]
        interp_y[singles_y[:, 0, 0], singles_y[:, 0, 2], singles_y[:, 0, 1], 0] = data_cut[singles_y[:, 0, 0], singles_y[:, 0, 2], singles_y[:, 0, 1], 0]

        # Fill with mask with ones and restore value where value was written on bigger
        interp_x = interp_x + mask[:, 1:st_sz[0]-1, 1:st_sz[1]-1, :]
        interp_x = np.where(interp_x > 1, interp_x-1, interp_x)
        interp_y = interp_y + mask[:, 1:st_sz[0]-1, 1:st_sz[1]-1, :]
        interp_y = np.where(interp_y > 1, interp_y-1, interp_y)

        '''
        ind = 35
        pdat2 = data[ind].reshape((st_sz[0], st_sz[1])).copy()
        print(f'data:\n{pdat2}')
        pdat2 = mask[ind].reshape((st_sz[0], st_sz[1])).copy()
        print(f'mask:\n{pdat2}')
        pdat2 = data_cut[ind].reshape((st_sz[0]-2, st_sz[1]-2)).copy()
        print(f'data_cut:\n{pdat2}')
        pdat2 = mask_x[ind].reshape((st_sz[0]-2, st_sz[1]-2)).copy()
        print(f'mask_x:\n{pdat2}')
        pdat2 = interp_x[ind].reshape((st_sz[0]-2, st_sz[1]-2)).copy()
        print(f'interp_x:\n{pdat2}')
        pdat2 = mask_y[ind].reshape((st_sz[0]-2, st_sz[1]-2)).copy()
        print(f'mask_y:\n{pdat2}')
        pdat2 = interp_y[ind].reshape((st_sz[0]-2, st_sz[1]-2)).copy()
        print(f'interp_y:\n{pdat2}')
        # '''
        
        # Flatten interp_x and interp_y
        interp_x = np.reshape (interp_x, (interp_x.shape[0], np.prod(interp_x.shape[1:3])))
        interp_y = np.reshape (interp_y, (interp_y.shape[0], np.prod(interp_y.shape[1:3])))

        # interp_x und interp_y sind wahrscheinlich vertauscht


        print('Teststencil output')

        pdat2 = teststencil_x.reshape((st_sz[0], st_sz[1]))[1:st_sz[0]-1, 1:st_sz[1]-1].copy()
        print(f'T1x input:\n{pdat2}')
        ind = -4
        pdat2 = interp_x[ind].reshape((st_sz[0]-2, st_sz[1]-2)).copy()
        print(f'T1x interp_x:\n{pdat2}')
        #'''
        pdat2 = teststencil_y.reshape((st_sz[0], st_sz[1]))[1:st_sz[0]-1, 1:st_sz[1]-1].copy()
        # print(f'T1y input:\n{pdat2}')
        ind = -3
        pdat2 = interp_y[ind].reshape((st_sz[0]-2, st_sz[1]-2)).copy()
        pdat2 = np.rot90(pdat2,k=-1)
        print(f'T1y interp_y:\n{pdat2}')

        pdat2 = teststencil_x_rot.reshape((st_sz[0], st_sz[1]))[1:st_sz[0]-1, 1:st_sz[1]-1].copy()
        # print(f'T2x input:\n{pdat2}')
        ind = -2
        pdat2 = interp_x[ind].reshape((st_sz[0]-2, st_sz[1]-2)).copy()
        pdat2 = np.rot90(pdat2,k=-2)
        print(f'T2x interp_x:\n{pdat2}')

        pdat2 = teststencil_y_rot.reshape((st_sz[0], st_sz[1]))[1:st_sz[0]-1, 1:st_sz[1]-1].copy()
        # print(f'T2y input:\n{pdat2}')
        ind = -1
        pdat2 = interp_y[ind].reshape((st_sz[0]-2, st_sz[1]-2)).copy()
        pdat2 = np.rot90(pdat2,k=-3)
        print(f'T2y interp_y:\n{pdat2}')
        # '''


        # Glue them together (shape is data_length x 50 for 7x7 stencil, first 25 are x, second are y)
        data = np.concatenate((interp_x, interp_y), axis=1)

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
        labels = dataset[0].copy()
        # Get shape of data
        shape = data.shape
        # Check if data was transformed (shape = 4) or not (shape = 2), reshape data that was not transformed
        if len(shape) == 2:
            data = np.reshape(data, (shape[0], st_sz[0], st_sz[1], 1))

        # Was hier gemacht werden muss: 1. dynamisch an input angepasst 2. Glättungen eingefügt
        # Find midpoint
        i = int((st_sz[0]-1)/2)
        j = int((st_sz[1]-1)/2)
        ds = data.shape

        # Smoothing of data
        # weights
        w = np.array([-1, 4, 10])
        w4 = 2*w[0]+2*w[1]+w[2]
        w = w/w4

        kernel = np.array([[   0,    0,   w[0],    0,    0],
                           [   0,    0,   w[1],    0,    0],
                           [w[0], w[1], 2*w[2], w[1], w[0]],
                           [   0,    0,   w[1],    0,    0],
                           [   0,    0,   w[0],    0,    0]])/2
        kernel = np.tile(kernel, (ds[0], 1, 1, 1)).transpose((0, 2, 3, 1))

        data_sm = np.zeros((ds[0], ds[1]-4, ds[2]-4, ds[3]))
        for x in range(2, ds[1]-2):
            for y in range(2, ds[2]-2):
                data_sm[:, y-2, x-2, :] = np.sum(np.multiply(kernel, data[:, y-2:y+3, x-2:x+3, :]), axis=(1, 2))

        dss = data_sm.shape

        # 1st derivative
        c_dy = np.zeros((dss[0], dss[1]-4, dss[2]-4, dss[3]))  # Eig. nur -2 nötig, aber Rand wird für kappa eh nicht gebraucht
        c_dx = np.zeros((dss[0], dss[1]-4, dss[2]-4, dss[3]))
        for x in range(2, dss[2]-2):
            for y in range(2, dss[1]-2):
                c_dy[:, y-2, x-2, 0] = (data_sm[:, y+1, x, 0] - data_sm[:, y-1, x, 0])/2
                c_dx[:, y-2, x-2, 0] = (data_sm[:, y, x+1, 0] - data_sm[:, y, x-1, 0])/2


        # 2nd derivative
        c_dyy = np.zeros((dss[0], dss[1]-4, dss[2]-4, dss[3]))
        c_dxx = np.zeros((dss[0], dss[1]-4, dss[2]-4, dss[3]))
        c_dxy = np.zeros((dss[0], dss[1]-4, dss[2]-4, dss[3]))
        for x in range(2, dss[2]-2):
            for y in range(2, dss[1]-2):
                c_dxx[:, y-2, x-2, 0] = (data_sm[:, y+2, x, 0] - 2*data_sm[:, y, x, 0] + data_sm[:, y-2, x, 0])/4
                c_dyy[:, y-2, x-2, 0] = (data_sm[:, y, x+2, 0] - 2*data_sm[:, y, x, 0] + data_sm[:, y, x-2, 0])/4
                c_dxy[:, y-2, x-2, 0] = (data_sm[:, y+1, x+1, 0] - data_sm[:, y-1, x+1, 0] - data_sm[:, y+1, x-1, 0] + data_sm[:, y-1, x-1, 0])/4

        cds = c_dxx.shape

        kappa = np.zeros((cds[0], cds[1], cds[2], cds[3]))
        for x in range(0, cds[2]):
            for y in range(0, cds[1]):
                kappa[:, y, x, :] = 2*np.divide(
                    np.multiply(c_dxx[:, y, x, :], np.multiply(c_dx[:, y, x, :], c_dx[:, y, x, :])) -
                    np.multiply(c_dxy[:, y, x, :], np.multiply(c_dx[:, y, x, :], c_dy[:, y, x, :]))*2 +
                    np.multiply(c_dyy[:, y, x, :], np.multiply(c_dy[:, y, x, :], c_dy[:, y, x, :]))
                    ,
                    (np.multiply(c_dx[:, y, x, :], c_dx[:, y, x, :]) + np.multiply(c_dy[:, y, x, :], c_dy[:, y, x, :]))**(3/2)
                )

        kappa = np.nan_to_num(kappa)

        '''
        values = (np.multiply(c_dx[:, y, x, :], c_dx[:, y, x, :]) + np.multiply(c_dy[:, y, x, :], c_dy[:, y, x, :]))**(3/2)
        values = np.where(values == 0)
        print(f'values:\n{values}')
        # '''

        ks = kappa.shape

        # Smoothing of kappa
        # weights
        w = np.array([1, 4, 6])
        w4 = 2*w[0]+2*w[1]+w[2]
        w = w/w4

        kernel = np.array([[   0,    0,   w[0],    0,    0],
                           [   0,    0,   w[1],    0,    0],
                           [w[0], w[1], 2*w[2], w[1], w[0]],
                           [   0,    0,   w[1],    0,    0],
                           [   0,    0,   w[0],    0,    0]])/2
        kernel = np.tile(kernel, (ks[0], 1, 1, 1)).transpose((0, 2, 3, 1))

        kappa_sm = np.zeros((ks[0], ks[1]-4, ks[2]-4, ks[3]))
        for x in range(2, ks[1]-2):
            for y in range(2, ks[2]-2):
                kappa_sm[:, y-2, x-2, :] = np.sum(np.multiply(kernel, kappa[:, y-2:y+3, x-2:x+3, :]), axis=(1, 2))
        # kappa_sm = kappa_sm[:, 0, 0, :]
        j = int((data_sm.shape[1]-1)/2)
        i = int((data_sm.shape[2]-1)/2)

        weights = 1-2*np.abs(0.5 - data_sm[:, j-1:j+2, i-1:i+2, :])

        kappa_out = np.divide(
            np.sum(np.multiply(weights, kappa_sm), axis=(1, 2))
            ,
            np.sum(weights, axis=(1, 2))
        )
        print(f'kappa_out.shape:\n{kappa_out.shape}')

        '''
        # Test
        ind = 95912
        datapoint = data[ind, 0, 5, 0]
        print(f'datapoint:\n{datapoint}')
        print_data_grad = data.transpose((0, 1, 3, 2))[ind]
        print(f'\nData:\n{print_data_grad}')
        print_data_sm = data_sm.transpose((0, 1, 3, 2))[ind]
        print(f'print_data_sm:\n{print_data_sm}')
        print_kappa = kappa.transpose((0, 1, 3, 2))[ind]
        print(f'print_kappa:\n{print_kappa}')
        print_kappa_sm = kappa_sm[ind]
        print(f'print_kappa_sm:\n{print_kappa_sm}')
        label = labels[ind]
        print(f'label:\n{label}')
        # '''

        '''
        prt_labels = labels[:10]
        prt_kappas = kappa_sm[:10]
        print(f'prt_labels:\n{prt_labels}')
        print(f'prt_kappas:\n{prt_kappas}')
        # '''
        # Reshape to tensor if angle matrix is needed, otherwise just output vectors
        if (data.shape != shape) & (self.parameters['angle']):
            # Reshape transformed data to original shape
            data = np.reshape(data, shape)

        return [dataset[0], dataset[1], kappa_out]


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
        kappa = -np.round(
            2/Delta*h_xx/((1+np.multiply(h_x, h_x))**(3/2))
            , 5)

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
