import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from scipy.ndimage.morphology import distance_transform_edt as edt
import numpy as np
import cv2
from scipy.ndimage import convolve


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

        lenth = torch.mean(torch.sqrt(delta_u + self.smooth)
                           )  # equ.(11) in the paper

        """
        region term
        """

        C_1 = torch.ones((512, 512)).to(device="cuda:0")
        C_2 = torch.zeros((512, 512)).to(device="cuda:0")

        # equ.(12) in the paper
        region_in = torch.abs(
            torch.mean(pred[:, 0, :, :] * ((target[:, 0, :, :] - C_1)**2)))
        # equ.(12) in the paper
        region_out = torch.abs(
            torch.mean((1-pred[:, 0, :, :]) * ((target[:, 0, :, :] - C_2)**2)))

        lambdaP = 1  # lambda parameter could be various.
        mu = 1  # mu parameter could be various.
        with open('debug.txt', '+a') as f:
            f.write(str(region_in) + ' ' + str(region_out) + '\n')
            f.write(str(lenth) + '\n')
        f.close()

        return lenth + lambdaP * (mu * region_in + region_out)


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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, multiclass=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.multiclass = multiclass

    def dice_score(self, inputs, targets):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_score = (2.*intersection + self.smooth) / \
            (inputs.sum() + targets.sum() + self.smooth)
        return dice_score

    def dice_score_mc(self, inputs, targets):
        inputs = F.softmax(inputs)
        # # for 4 class
        # targets = F.one_hot(
        #     targets.long(), 4).permute(0, 3, 1, 2)
        # for 5 class
        targets = F.one_hot(
            targets.long(), 5).permute(0, 3, 1, 2)
        inputs = inputs.flatten(0, 1)
        targets = targets.flatten(0, 1)
        intersection = (inputs * targets).sum()
        dice_score = (2.*intersection + self.smooth) / \
            (inputs.sum() + targets.sum() + self.smooth)
        return dice_score

    def forward(self, inputs, targets):
        if self.multiclass:
            dice_score = self.dice_score_mc(inputs, targets)
        else:
            dice_score = self.dice_score(inputs, targets)
        return 1 - dice_score


class MultitaskUncertaintyLoss(nn.Module):
    def __init__(self):
        super(MultitaskUncertaintyLoss, self).__init__()

    # def forward(self, loss_values, log_var_tasks):
    #     total_loss = 0
    #     loss_cls, loss_reg = loss_values
    #     log_var_task1, log_var_task2 = log_var_tasks
    #     total_loss += (loss_cls.cpu() /
    #                    (2*torch.exp(log_var_task1))) + log_var_task1
    #     total_loss += (loss_reg.cpu() / torch.exp(2 *
    #                    log_var_task2)) + log_var_task2

    #     return torch.mean(total_loss)

    def forward(self, loss_values, log_var_tasks, regg_flag):
        total_loss = 0
        for i in range(0, len(loss_values)):
            dtype = loss_values[i].dtype
            device = loss_values[i].device
            stds = (
                torch.exp(log_var_tasks[i])**(1/2)).to(device).to(dtype)
            if regg_flag[i]:
                coeff = 1 / (2*(stds**2))
            else:
                coeff = 1 / (stds**2)
            total_loss += coeff*loss_values[i] + torch.log(stds)
        return total_loss

    # # with sigma
    # def forward(self, loss_values, sigmas_sqr):
    #     total_loss = 0
    #     for i in range(0, len(loss_values)):
    #         total_loss += (loss_values[i].cpu() /
    #                        (2*sigmas_sqr[i])) + torch.log(sigmas_sqr[i])
    #     return total_loss

    # def forward(self, loss_values, log_var_tasks):
    #     total_loss = 0
    #     for i in range(0, len(loss_values)):
    #         total_loss += loss_values[i].cpu() * \
    #             torch.exp(-log_var_tasks[i]) + log_var_tasks[i]
    #     return total_loss/4


class MultiTaskLoss(torch.nn.Module):
    '''https://arxiv.org/abs/1705.07115'''

    def __init__(self, is_regression, reduction='none'):
        super(MultiTaskLoss, self).__init__()
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, losses):
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)
        coeffs = 1 / ((self.is_regression+1)*(stds**2))
        multi_task_losses = coeffs*losses + torch.log(stds)
        # print(torch.log(stds))
        # print(coeffs)
        # print(coeffs*losses)

        if self.reduction == 'sum':
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == 'mean':
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses


def calc_loss(pred, target, bce_weight=0.5, loss_type='mse'):
    if loss_type == 'BCE':
        loss = nn.BCEWithLogitsLoss()(pred, target)
    if loss_type == 'ce':
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
        loss = DiceLoss()(pred, target) + nn.BCEWithLogitsLoss()(pred, target)
    if loss_type == 'dice_bce_mc':
        loss = DiceLoss(multiclass=True)(pred, target) + \
            nn.CrossEntropyLoss()(pred, target.long())
    if loss_type == 'dice_score':
        loss = DiceLoss().dice_score(pred, target)
    if loss_type == 'log_cosh_dice_loss':
        x = DiceLoss()(pred, target)
        loss = torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)
    if loss_type == 'dice_score_mc':
        loss = DiceLoss(multiclass=True).dice_score_mc(pred, target)
    if loss_type == 'HausdorffDTLoss':
        loss = HausdorffDTLoss()(pred, target, debug=False)
    if loss_type == "ActiveContourLoss":
        loss = ActiveContourLoss()(pred, target)
    return loss
