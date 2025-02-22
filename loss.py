import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from scipy.ndimage.morphology import distance_transform_edt as edt
import numpy as np
import cv2
from scipy.ndimage import convolve
from topoloss_pytorch import getTopoLoss
from topoCount import topoCountloss
from myTopoLoss import PointCloudFiltration
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


class MultitaskUncertaintyLoss(nn.Module):
    def __init__(self):
        super(MultitaskUncertaintyLoss, self).__init__()

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2, logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        #pt -> true probability
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False) #-log(pt)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss #0.25* (1-pt)^gamma *-log(pt)
        return torch.mean(F_loss)

def flatten(input, target):
    num_class = input.size(1)
    #B, C, H, W -> # B, H, W, C 
    input = input.permute(0, 2, 3, 1).contiguous()
    
    input_flatten = input.view(-1, num_class)
    target_flatten = target.view(-1)
        
    return input_flatten, target_flatten 

class TopKLoss(nn.BCEWithLogitsLoss):
    def __init__(self, topk=2):
        super().__init__(reduction='none')
        self.topk = topk
    def forward(self, pred, target):
        pred, target = flatten(pred, target)
        pred = pred[:, 0]
        #foreground prob
        foreground_prob = F.sigmoid(pred)
        #backround prob
        background_prob = 1-foreground_prob
        #select ground-truth probability
        input_prob = torch.gather(torch.stack((background_prob, foreground_prob), dim=1), 1, target.unsqueeze(1).long())
        input_prob = input_prob[:,0]
        # select indices of lowest probabilities
        values, indices = torch.topk(input_prob, len(target)//self.topk, largest=False)
        #BCE loss with logits
        cross_entropy = super().forward(pred, target)

        # Create a mask
        mask = torch.zeros_like(cross_entropy)  
        mask[indices] = 1  # Set True for the indices to consider

        loss = cross_entropy[mask > 0].mean()  # Average only the selected losses
        return loss

class FocalTverskyLoss(nn.Module):
    # When α = β = 0.5 it reduces to the Dice coefficient
    # and when α = β = 1 it reduces to the IoU.
    def __init__(self, smooth=1.0, alpha=0.5, beta=0.5, gamma=1):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        pred, target = flatten(pred, target)

        # target = B*H*W
        # pred = B*H*W, C
        num_classes = pred.size(1)
        if num_classes==1:
            pred = pred[:, 0]
            pred = F.sigmoid(pred)
            t_p = (pred * target).sum()
            f_p = ((1-target) * pred).sum()
            f_n = (target * (1-pred)).sum()
            tversky = (t_p + self.smooth) / (t_p + self.alpha*f_p + self.beta*f_n + self.smooth)
            loss = (1 - tversky)**self.gamma
        else:
            pred = F.softmax(pred, dim=1)
            losses = []
            for c in range(num_classes):
                target_c = (target == c).float()
                input_c = pred[:, c]
                
                t_p = (input_c * target_c).sum()
                f_p = ((1-target_c) * input_c).sum()
                f_n = (target_c * (1-input_c)).sum()
                tversky = (t_p + self.smooth) / (t_p + self.alpha*f_p + self.beta*f_n + self.smooth)
                focal_tversky = (1 - tversky)**self.gamma
                losses.append(focal_tversky)
            
            losses = torch.stack(losses)
            loss = losses.mean()
        return loss

def calc_loss(pred, target, bce_weight=0.5, loss_type='mse'):
    if loss_type == 'BCE':
        loss = nn.BCEWithLogitsLoss()(pred.squeeze(1), target)
    if loss_type == 'TopK':
        loss = TopKLoss()(pred, target)
    if loss_type == 'MyTopoLoss1':
        lamda_pers = 0.001
        loss_dice = DiceLoss(CLASS_NUMBER)(pred, target, softmax=True)
        loss_ce = nn.CrossEntropyLoss()(pred, target[:].long())
        # Step 1: Apply softmax to convert logits into probabilities
        # Softmax is applied along the class dimension (dim=1)
        likelihood = F.softmax(pred, dim=1)  # Shape: (B, 3, H, W)
        out = torch.argmax(likelihood, dim=1).cpu().detach().numpy()
        
        # Assume ground_truth_immune and ground_truth_cells are the binary masks for the immune and cell classes
        ground_truth_cells = (target == 1).cpu().numpy().astype(np.uint8)   # Convert cell mask (class 1) to binary
        ground_truth_immune = (target == 2).cpu().numpy().astype(np.uint8) # Convert immune mask (class 2) to binary
        
        total_loss_pers = 0
        for batch in range(out.shape[0]):
            
            # Assume ground_truth_immune and ground_truth_cells are the binary masks for the immune and cell classes
            pred_cells = (out[batch] == 1).astype(np.uint8)  # Convert cell mask (class 1) to binary
            pred_immune = (out[batch] == 2).astype(np.uint8)  # Convert immune mask (class 2) to binary
            total_loss_pers += PointCloudFiltration(ground_truth_cells[batch], pred_cells, 'betti',p=2)
            total_loss_pers += PointCloudFiltration(ground_truth_immune[batch], pred_immune, 'betti',p=2)
        total_loss_pers /= out.shape[0]
        total_loss_pers = torch.tensor(total_loss_pers, dtype=torch.float32, device=pred.device)

        loss = (0.5 * loss_ce + 0.5 * loss_dice)*(1+(lamda_pers*total_loss_pers))
        
    if loss_type == 'TopoCount':
        lamda_pers = 1
        loss_dice = DiceLoss(CLASS_NUMBER)(pred, target, softmax=True)
        loss_ce = nn.CrossEntropyLoss()(pred, target[:].long())
        # Step 1: Apply softmax to convert logits into probabilities
        # Softmax is applied along the class dimension (dim=1)
        likelihood = F.softmax(pred, dim=1)  # Shape: (B, 3, H, W)
        
        # Step 2: Extract the probabilities for the two classes of interest
        # Class 1: Immune (index 1 in the channel dimension)
        likelihood_cells  = likelihood[:, 1, :, :]  # Shape: (B, H, W)
        
        # Class 2: Cells (index 2 in the channel dimension)
        likelihood_immune = likelihood[:, 2, :, :]  # Shape: (B, H, W)
        
        # Assume ground_truth_immune and ground_truth_cells are the binary masks for the immune and cell classes
        ground_truth_cells = (target == 1).float()   # Convert cell mask (class 1) to binary
        ground_truth_immune = (target == 2).float()  # Convert immune mask (class 2) to binary
        
        # Compute topological loss for immune and cell classes
        topo_loss_cells = topoCountloss(likelihood_cells.unsqueeze(1), ground_truth_cells.unsqueeze(1))
        topo_loss_immune = topoCountloss(likelihood_immune.unsqueeze(1), ground_truth_immune.unsqueeze(1))

        loss = (loss_dice+loss_ce) +(lamda_pers * (topo_loss_immune+topo_loss_cells)) 
    if loss_type == 'TopoCount2':
        lamda_pers = 1
        loss_dice = DiceLoss(CLASS_NUMBER)(pred, target, softmax=True)
        loss_ce = nn.CrossEntropyLoss()(pred, target[:].long())
        # Step 1: Apply softmax to convert logits into probabilities
        # Softmax is applied along the class dimension (dim=1)
        likelihoods = F.softmax(pred, dim=1)  # Shape: (B, 3, H, W)
        
        
        ground_truth_foreground = (target != 0).float()  
        print(torch.unique(ground_truth_foreground))
        likelihood_foreground = likelihoods[:, 1, :, :] + likelihoods[:, 2, :, :]  # Shape: (B, H, W), combined foreground class
        
        # Compute topological loss for immune and cell classes
        topo_loss = topoCountloss(likelihood_foreground.unsqueeze(1), ground_truth_foreground.unsqueeze(1))

        loss = (loss_dice+loss_ce) + lamda_pers * topo_loss 
    if loss_type == 'TopoLoss':
        lamda_pers = 0.0001
        loss_dice = DiceLoss(CLASS_NUMBER)(pred, target, softmax=True)
        loss_ce = nn.CrossEntropyLoss()(pred, target[:].long())
        # Step 1: Apply softmax to convert logits into probabilities
        # Softmax is applied along the class dimension (dim=1)
        likelihood = F.softmax(pred, dim=1)  # Shape: (B, 3, H, W)
        
        # Step 2: Extract the probabilities for the two classes of interest
        # Class 1: Immune (index 1 in the channel dimension)
        likelihood_cells  = likelihood[:, 1, :, :]  # Shape: (B, H, W)
        
        # Class 2: Cells (index 2 in the channel dimension)
        likelihood_immune = likelihood[:, 2, :, :]  # Shape: (B, H, W)
        
        # Assume ground_truth_immune and ground_truth_cells are the binary masks for the immune and cell classes
        ground_truth_cells = (target == 1).float()   # Convert cell mask (class 1) to binary
        ground_truth_immune = (target == 2).float()  # Convert immune mask (class 2) to binary
        # Compute topological loss for immune and cell classes
        topo_loss_cells = getTopoLoss(likelihood_cells, ground_truth_cells)
        topo_loss_immune = getTopoLoss(likelihood_immune, ground_truth_immune)

        loss = (loss_dice+loss_ce) +(lamda_pers * (topo_loss_immune+topo_loss_cells)) 

    if loss_type == 'TopoLoss2':
        lamda_pers = 0.00005
        loss_dice = DiceLoss(CLASS_NUMBER)(pred, target, softmax=True)
        loss_ce = nn.CrossEntropyLoss()(pred, target[:].long())
        # Step 1: Apply softmax to convert logits into probabilities
        # Softmax is applied along the class dimension (dim=1)
        likelihoods = F.softmax(pred, dim=1)  # Shape: (B, 3, H, W)
        
        # Ground-truth for foreground vs background
        ground_truth_foreground = (target != 0).float()  
        # Ground-truth for foreground vs background
        likelihood_foreground = likelihoods[:, 1, :, :] + likelihoods[:, 2, :, :]  # Shape: (B, H, W), combined foreground class
        
        # Compute topological loss for immune and cell classes
        topo_loss = getTopoLoss(likelihood_foreground, ground_truth_foreground)

        loss = (loss_dice+loss_ce) + lamda_pers * topo_loss 

    if loss_type == 'BCE_HEM':
        batchBase = False
        loss = nn.BCEWithLogitsLoss(reduction='none')(pred.squeeze(1), target)
        #loss = torch.mean(loss)i
        if batchBase:
            loss = torch.mean(loss, axis=(1,2))
            # Select the top 4 losses
            topk_losses, topk_indices = torch.topk(loss, 2)
            mask = torch.zeros_like(loss)
            mask[topk_indices] = 1
            # Apply the mask to losses to keep only the selected ones
            masked_loss = (loss * mask).sum() / mask.sum()  # Mean of selected losses
            
        else:
            loss_f = loss.flatten()
            topk_losses, topk_indices = torch.topk(loss_f, 500)
            mask = torch.zeros_like(loss_f)
            mask[topk_indices] = 1
            # Apply the mask to losses to keep only the selected ones
            masked_loss = (loss_f * mask).sum() / mask.sum()  # Mean of selected losses
        loss = masked_loss
    if loss_type == 'CE':
        loss = nn.CrossEntropyLoss()(pred, target[:].long())
    if loss_type == 'FL':
        #loss = FocalLoss()(pred, target)
        loss = BinaryFocalLoss(gamma=2)(pred, target)
    if loss_type == 'mse':
        loss = nn.MSELoss()(pred.squeeze(1), target)
    if loss_type == 'mseMC':
        loss = nn.MSELoss()(pred, target)
    if loss_type == 'rmse':
        mse = nn.MSELoss()(pred, target)
        loss = torch.sqrt(mse)
    if loss_type == 'l1loss':
        loss = nn.L1Loss()(pred, target)
    if loss_type == 'dice':
        loss = DiceLoss()(pred, target)
    if loss_type == 'dice_bce':
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
    if loss_type == "Tversky":
        loss = FocalTverskyLoss(alpha=0.4, beta=0.6)(pred, target) 
    return loss
