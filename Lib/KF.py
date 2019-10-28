import numpy as np
import matplotlib.pyplot as plt

def Kalman_Filter(measure, sigmoid, SDT = []):
    """Kalman Filter

    Kalman Filter for turbo engine FDI.

    Args:
        measure: Sensor detection results with measurement errors,shape: n * length, type: np.array
        sigmoid: Measurement errors,shape: n * 1
        SDT: Small deviation table,shape: n * 8

    Returns:
        A sequential result,shape:n * length

    Raises:
        IOError: Size dismatch.
    """
    z_mat = measure
    if len(z_mat.shape) == 1:
        length = z_mat.shape[0]
        n_z = 1
        z_mat.resize((1, length))
    else:
        [n_z, length] = z_mat.shape
    if not SDT:
        SDT = np.eye(n_z)
        n_x = n_z
    else:
        n_x = z_mat.shape[1]
    x_mat = np.zeros((n_x,1))
    # 定义初始状态协方差矩阵
    p_mat = np.eye(n_z)
    # 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
    f_mat = np.eye(n_z)
    # 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
    q_mat = np.zeros((n_z, n_z))
    # 定义观测矩阵
    h_mat = SDT
    # 定义观测噪声协方差
    r_mat = sigmoid

    for i in range(length):
        x_predict = f_mat * x_mat
        p_predict = f_mat * p_mat * f_mat.T + q_mat
        kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
        x_mat = x_predict + kalman * (z_mat[:, i] - h_mat * x_predict)
        p_mat = (np.eye(n_z) - kalman * h_mat) * p_predict
        print(x_mat)
        # plt.plot(x_mat[0, 0], x_mat[1, 0], 'ro', markersize=1)

    plt.show()

if __name__ == '__main__':
    sigmoid = [[0.1]*1]
    foo = 1
    x = np.array([foo + np.random.normal() * 0.03 for _ in range(1000)])
    Kalman_Filter(x, sigmoid, [])

