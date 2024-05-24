import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from scipy.ndimage.morphology import distance_transform_edt as edt
import numpy as np
import cv2
from scipy.ndimage import convolve
global CLASS_NUMBER

class ActiveContourLoss(nn.Module):
    def __init__(self, smooth=0.00000001):
        super(ActiveContourLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # horizontal and vertical directions
        x = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        y = pred[:, :, :, 1:] - pred[:, :, :, :-1]

        delta_x = x[:, :, 1:, :-2]**2
        delta_y = y[:, :, :-2, 1:]**2
        delta_u = torch.abs(delta_x + delta_y)

        lenth = torch.sum(torch.sqrt(delta_u + self.smooth)
                          )  # equ.(11) in the paper

        """
        region term
        """

        C_1 = torch.ones((512, 512)).to(device="cuda:0")
        C_2 = torch.zeros((512, 512)).to(device="cuda:0")

        # equ.(12) in the paper
        region_in = torch.abs(
            torch.sum(pred[:, 0, :, :] * ((target[:, 0, :, :] - C_1)**2)))
        # equ.(12) in the paper
        region_out = torch.abs(
            torch.sum((1-pred[:, 0, :, :]) * ((target[:, 0, :, :] - C_2)**2)))

        lambdaP = 1  # lambda parameter could be various.
        mu = 1  # mu parameter could be various.

        return lenth + lambdaP * (mu * region_in + region_out)


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(
                    bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.cpu().numpy(),
                                     target.cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=0.2, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(
            self.distance_field(pred.detach().cpu().numpy())).float()
        target_dt = torch.from_numpy(
            self.distance_field(target.detach().cpu().numpy())).float()

        # pred_dt.requires_grad_(True)
        # target_dt.requires_grad_(True)

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance.to("cuda:0")
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6, multiclass=False):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.multiclass = multiclass

#     def dice_score(self, inputs, targets):
#         inputs = F.sigmoid(inputs)

#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         intersection = (inputs * targets).sum()
#         dice_score = (2.*intersection + self.smooth) / \
#             (inputs.sum() + targets.sum() + self.smooth)
#         return dice_score

#     def dice_score_mc(self, inputs, targets):
#         inputs = F.softmax(inputs)
#         targets = F.one_hot(
#             targets.long(), 4).permute(0, 3, 1, 2)
#         inputs = inputs.flatten(0, 1)
#         targets = targets.flatten(0, 1)
#         intersection = (inputs * targets).sum()
#         dice_score = (2.*intersection + self.smooth) / \
#             (inputs.sum() + targets.sum() + self.smooth)
#         return dice_score

#     def forward(self, inputs, targets):
#         if self.multiclass:
#             dice_score = self.dice_score_mc(inputs, targets)
#         else:
#             dice_score = self.dice_score(inputs, targets)
#         return 1 - dice_score

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

# ce_loss = nn.CrossEntropyLoss()
# dice_loss = DiceLoss(4)


class MultitaskUncertaintyLoss(nn.Module):
    def __init__(self):
        super(MultitaskUncertaintyLoss, self).__init__()

    # def forward(self, loss_values, log_var_tasks):
    #     total_loss = 0
    #     loss_cls, loss_reg = loss_values
    #     log_var_task1, log_var_task2 = log_var_tasks
    #     total_loss += (loss_cls.cpu() / (2*torch.exp(2 * log_var_task1))) + log_var_task1
    #     total_loss += (loss_reg.cpu() / torch.exp(2 * log_var_task2)) + log_var_task2
    #     return total_loss

    # def forward(self, loss_values, log_var_tasks):
    #     total_loss = 0
    #     loss_cls, loss_reg = loss_values
    #     log_var_task1, log_var_task2 = log_var_tasks
    #     total_loss += loss_cls.cpu() * torch.exp(-log_var_task1) + log_var_task1
    #     total_loss += loss_reg.cpu() * torch.exp(-log_var_task2) + log_var_task2
    #     return total_loss/2

    def forward(self, loss_values, log_var_tasks):
        total_loss = 0
        for i in range(0, len(loss_values)):
            total_loss += loss_values[i].cpu() * \
                torch.exp(-log_var_tasks[i]) + log_var_tasks[i]
        return total_loss/len(loss_values)


def calc_loss(pred, target, bce_weight=0.5, loss_type='mse'):
    if loss_type == 'BCE':
        loss = nn.BCEWithLogitsLoss()(pred, target)
    if loss_type == 'CE':
        loss = nn.CrossEntropyLoss()(pred, target.long())
    if loss_type == 'mse':
        loss = nn.MSELoss()(pred, target)
    if loss_type == 'rmse':
        mse = nn.MSELoss()(pred, target)
        loss = torch.sqrt(mse)
    if loss_type == 'l1loss':
        loss = nn.L1Loss()(pred, target)
    if loss_type == 'dice':
        loss = DiceLoss()(pred, target)
    if loss_type == 'dice_bce':
        #loss = DiceLoss()(pred, target) + nn.BCEWithLogitsLoss()(pred, target)

        loss_bce = nn.BCEWithLogitsLoss()(pred.squeeze(1), target)
        loss_dice = DiceLoss(CLASS_NUMBER)(pred, target, softmax=True)
        
        loss = 0.5 * loss_bce + 0.5 * loss_dice
        
    if loss_type == 'dice_bce_mc':
        #Method1
        # loss = DiceLoss(multiclass=True)(pred, target) + \
        #     nn.CrossEntropyLoss()(pred, target.long())
        
        #Method2.1
        # loss_ce = ce_loss(pred, target[:].long())
        # loss_dice = dice_loss(pred, target, softmax=True)
        
        #Method2.2
        loss_ce = nn.CrossEntropyLoss()(pred, target[:].long())
        loss_dice = DiceLoss(CLASS_NUMBER)(pred, target, softmax=True)
        
        loss = 0.5 * loss_ce + 0.5 * loss_dice
    if loss_type == 'dice_score':
        loss = DiceLoss().dice_score(pred, target)
    if loss_type == 'log_cosh_dice_loss':
        x = DiceLoss()(pred, target)
        loss = torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)
    if loss_type == 'dice_score_mc':
        loss = DiceLoss(multiclass=True).dice_score_mc(pred, target)
    if loss_type == 'HausdorffDTLoss':
        loss = HausdorffDTLoss()(pred, target, debug=False)
    if loss_type == 'HausdorffERLoss':
        loss = HausdorffERLoss()(pred, target, debug=False)
    if loss_type == "ActiveContourLoss":
        loss = ActiveContourLoss()(pred, target)
    return loss
