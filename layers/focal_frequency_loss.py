import torch
import torch.nn as nn

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # Assuming x is a 1D tensor
        if IS_HIGH_VERSION:
            freq = torch.fft.fft(x, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(x, 1, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_vec, real_vec, matrix=None):
    # Frequency distance using (squared) Euclidean distance
        freq_distance = torch.sum((recon_vec - real_vec) ** 2)/1536
        # print("freq_distance",freq_distance)

    # Spectrum weight matrix
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
        # If the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = torch.sqrt(freq_distance) ** self.alpha
            # print("matrix_tmp",matrix_tmp)

        # Whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

        # Whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max()

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()
            # print("weight_matrix",weight_matrix)

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
        'The values of spectrum weight matrix should be in the range [0, 1], '
        'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

    # Dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        # print('loss',loss)
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        # print("pred",pred)
        # print("target",target)
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (batch_size, channels). Predicted tensor.
            target (torch.Tensor): of shape (batch_size, channels). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        
        pred_freq = torch.nn.functional.normalize(pred_freq, dim=-1)
        target_freq = torch.nn.functional.normalize(target_freq, dim=-1)
        # print(pred_freq)
        # print(target_freq)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight