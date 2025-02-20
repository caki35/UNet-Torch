import torch
from scipy import misc
import numpy as np
import math
import torch.nn as nn
import random 
from PersistencePython import cubePers

topo_size = 50; # tiling patch size for persistence loss
sub_patch_border_width = 5
padwidth = 3;
mm=1
device = "cuda:0"

def compute_persistence_2DImg_1DHom_lh(f, padwidth = 2, homo_dim=1, pers_thd=0.001):
    """
    compute persistence diagram in a 2D function (can be N-dim) and critical pts
    only generate 1D homology dots and critical points
    """
    assert len(f.shape) == 2  # f has to be 2D function
    dim = 2

    # pad the function with a few pixels of minimum values
    # this way one can compute the 1D topology as loops
    # remember to transform back to the original coordinates when finished
    #padwidth = 2
    # padvalue = min(f.min(), 0.0)
    padvalue = f.min()
    # print (type(f.cpu().detach().numpy()))
    # print (padvalue)
    if(not isinstance(f, np.ndarray)):
        f_padded = np.pad(f.cpu().detach().numpy(),padwidth, 'constant', constant_values=padvalue.cpu().detach().numpy())
    else:
        f_padded = np.pad(f,padwidth, 'constant', constant_values=padvalue)


    # call persistence code to compute diagrams
    # loads PersistencePython.so (compiled from C++); should be in current dir
    #from src.PersistencePython import cubePers
    from PersistencePython import cubePers

    # persistence_result = cubePers(a, list(f_padded.shape), 0.001)
    persistence_result = cubePers(np.reshape(
        f_padded, f_padded.size).tolist(), list(f_padded.shape), pers_thd)

    # print("persistence_result", type(persistence_result))
    # print(type(persistence_result))
    # print (persistence_result)
    # print(len(persistence_result))

    # only take 1-dim topology, first column of persistence_result is dimension
    persistence_result_filtered = np.array(list(filter(lambda x: x[0] == homo_dim,
                                                  persistence_result)))

    # persistence diagram (second and third columns are coordinates)
    # print (persistence_result_filtered)
    if(persistence_result_filtered.shape[0]==0):
        return np.array([]), np.array([]), np.array([])
    dgm = persistence_result_filtered[:, 1:3]

    # critical points
    birth_cp_list = persistence_result_filtered[:, 4:4 + dim]
    death_cp_list = persistence_result_filtered[:, 4 + dim:]

    # when mapping back, shift critical points back to the original coordinates
    birth_cp_list = birth_cp_list - padwidth
    death_cp_list = death_cp_list - padwidth

    return dgm, birth_cp_list, death_cp_list

def compute_persistence_2DImg_1DHom_gt(f, padwidth = 2, homo_dim=1, pers_thd=0.001):
    """
    compute persistence diagram in a 2D function (can be N-dim) and critical pts
    only generate 1D homology dots and critical points
    """
    # print (len(f.shape))
    assert len(f.shape) == 2  # f has to be 2D function
    dim = 2

    # pad the function with a few pixels of minimum values
    # this way one can compute the 1D topology as loops
    # remember to transform back to the original coordinates when finished
    #padwidth = 2
    # padvalue = min(f.min(), 0.0)
    padvalue = f.min()
    # print(f)
    # print (type(f.cpu().numpy()))
    if(not isinstance(f, np.ndarray)):
        f_padded = np.pad(f.cpu().detach().numpy(),padwidth, 'constant', constant_values=padvalue.cpu().detach().numpy())
    else:
        f_padded = np.pad(f,padwidth, 'constant', constant_values=padvalue)

    # call persistence code to compute diagrams
    # loads PersistencePython.so (compiled from C++); should be in current dir
    #from src.PersistencePython import cubePers
    from PersistencePython import cubePers

    # persistence_result = cubePers(a, list(f_padded.shape), 0.001)
    persistence_result = cubePers(np.reshape(
        f_padded, f_padded.size).tolist(), list(f_padded.shape), pers_thd)

    # print("persistence_result", type(persistence_result))
    # print(type(persistence_result))
    # print (persistence_result)
    # print(len(persistence_result))

    # only take 1-dim topology, first column of persistence_result is dimension
    persistence_result_filtered = np.array(list(filter(lambda x: x[0] == homo_dim,
                                                  persistence_result)))

    # persistence diagram (second and third columns are coordinates)
    # print (persistence_result_filtered)
    #print ('shape of persistence_result_filtered')
    #print (persistence_result_filtered.shape)
    if(persistence_result_filtered.shape[0]==0):
        return np.array([]), np.array([]), np.array([])
    dgm = persistence_result_filtered[:, 1:3]

    # critical points
    birth_cp_list = persistence_result_filtered[:, 4:4 + dim]
    death_cp_list = persistence_result_filtered[:, 4 + dim:]

    # when mapping back, shift critical points back to the original coordinates
    birth_cp_list = birth_cp_list - padwidth
    death_cp_list = death_cp_list - padwidth

    return dgm, birth_cp_list, death_cp_list

def compute_dgm_force(lh_dgm, gt_dgm, pers_thresh=0.03, pers_thresh_perfect=0.99, do_return_perfect=False):
    # get persistence list from both diagrams
    lh_pers = abs(lh_dgm[:, 1] - lh_dgm[:, 0])
    if(gt_dgm.shape[0] == 0):
        gt_pers = None;
        gt_n_holes = 0;
    else:
        gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]
        gt_n_holes = gt_pers.size  # number of holes in gt
        #gt_n_holes = (gt_pers > 0.03).sum()  # number of holes in gt # ignore flat ones

    if(gt_pers is None or gt_n_holes == 0):
        idx_holes_to_fix = list();
        idx_holes_to_remove = list(set(range(lh_pers.size)))
        idx_holes_perfect = list();
    else:
        # more lh dots than gt dots

        #print (lh_pers.shape)
        #print(lh_pers.size)
        #print (gt_pers.shape)
        #assert lh_pers.size > gt_pers.size #?????????????????????????????????
        #assert lh_pers.size >= gt_n_holes #?????????????????????????????????
        if(lh_pers.size < gt_n_holes):
            gt_n_holes = lh_pers.size

        # check to ensure that all gt dots have persistence 1
        tmp = gt_pers > pers_thresh_perfect

        #assert tmp.sum() == gt_pers.size


        # get "perfect holes" - holes which do not need to be fixed, i.e., find top
        # lh_n_holes_perfect indices
        # check to ensure that at least one dot has persistence 1; it is the hole
        # formed by the padded boundary
        # if no hole is ~1 (ie >.999) then just take all holes with max values
        tmp = lh_pers > pers_thresh_perfect  # old: assert tmp.sum() >= 1
        # print('pers_thresh_perfect',pers_thresh_perfect)
        # print('lh_pers > pers_thresh_perfect',(lh_pers > pers_thresh_perfect).sum())
        #print (type(tmp))
        lh_pers_sorted_indices = np.argsort(lh_pers)[::-1]
        if np.sum(tmp) >= 1:
        # if tmp.sum >= 1:
            # n_holes_to_fix = gt_n_holes - lh_n_holes_perfect
            lh_n_holes_perfect = tmp.sum()
            #idx_holes_perfect = np.argpartition(lh_pers, -lh_n_holes_perfect)[
            #                    -lh_n_holes_perfect:]
            idx_holes_perfect = lh_pers_sorted_indices[:lh_n_holes_perfect];
        else:
            #idx_holes_perfect = np.where(lh_pers == lh_pers.max())[0]
            idx_holes_perfect = list();

        # find top gt_n_holes indices
        #idx_holes_to_fix_or_perfect = np.argpartition(lh_pers, -gt_n_holes)[
        #                              -gt_n_holes:]
        idx_holes_to_fix_or_perfect = lh_pers_sorted_indices[:gt_n_holes];


        # the difference is holes to be fixed to perfect
        idx_holes_to_fix = list(
            set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

        # remaining holes are all to be removed
        #idx_holes_to_remove = list(
        #    set(range(lh_pers.size)) - set(idx_holes_to_fix_or_perfect))
        idx_holes_to_remove = lh_pers_sorted_indices[gt_n_holes:];

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    # TODO values below this are small dents so dont fix them; tune this value?
    pers_thd = pers_thresh
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)
    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)

    if(do_return_perfect):
        return force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect
    
    return force_list, idx_holes_to_fix, idx_holes_to_remove

def topoCountloss(et_dmap, gt_dmap):    
    loss_pers = torch.tensor(0)
    n_fix = 0
    n_remove = 0
    topo_cp_weight_map = np.zeros(et_dmap.shape);
    topo_cp_weight_map_vis_b_fix = np.zeros(et_dmap.shape);
    topo_cp_weight_map_vis_d_fix = np.zeros(et_dmap.shape);
    topo_cp_weight_map_vis_b_rem = np.zeros(et_dmap.shape);
    topo_cp_weight_map_vis_d_rem = np.zeros(et_dmap.shape);
    topo_cp_weight_map_vis_b_gt = np.zeros(et_dmap.shape);
    topo_cp_weight_map_vis_d_gt = np.zeros(et_dmap.shape);
    topo_cp_weight_map_vis_b_perf = np.zeros(et_dmap.shape);
    topo_cp_weight_map_vis_d_perf = np.zeros(et_dmap.shape);
    topo_cp_weight_map_vis_grid = np.zeros(et_dmap.shape);
    topo_cp_ref_map = np.zeros(et_dmap.shape);

    gt_dmap_j = gt_dmap.detach().cpu().numpy();
    et_dmap_j = et_dmap.detach().cpu().numpy();
    min_x = max(10 , random.randint(0,int(topo_size/2)));
    min_y = max(10 , random.randint(0,int(topo_size/2)));
    max_x = -10 - topo_size;
    max_y = -10 - topo_size;
    et_sig = et_dmap.squeeze(dim=1)
    for y in range(min_y, gt_dmap_j.shape[-2]+max_y, topo_size-2*sub_patch_border_width):
        for x in range(min_x, gt_dmap_j.shape[-1]+max_x, topo_size-2*sub_patch_border_width):
            #if(random.randint(0,1)==1):
            #    continue 
            topo_cp_weight_map_vis_grid[0,0,y,x] = 1
            #print('y=',y)
            #print('x=',x)
            likelihood_sig = et_sig[:,y:min(y+topo_size, gt_dmap_j.shape[-2]), x:min(x+topo_size, gt_dmap_j.shape[-1])].squeeze();
            likelihood = likelihood_sig.detach().cpu().numpy();
            groundtruth = gt_dmap_j[:,:, y:min(y+topo_size, gt_dmap_j.shape[-2]), x:min(x+topo_size, gt_dmap_j.shape[-1])].squeeze();
        
            #print('likelihood.shape= ', likelihood.shape)
            #print('groundtruth.shape=', groundtruth.shape)
            if(len(likelihood.shape) < 2 or len(groundtruth.shape) < 2 ):
                continue;
            if(topo_size >= 100):
                likelihood_2 = misc.imresize(likelihood, (likelihood.shape[0]//2, likelihood.shape[1]//2)) 
                if(likelihood_2.max() > 0):
                    likelihood_2 = likelihood_2/likelihood_2.max()*likelihood.max()
                groundtruth_2 = misc.imresize(groundtruth, (groundtruth.shape[0]//2, groundtruth.shape[1]//2))
                if(groundtruth_2.max() > 0):
                    groundtruth_2 = groundtruth_2/groundtruth_2.max()*groundtruth.max()
                pd_lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom_lh(-likelihood_2*mm, padwidth = padwidth, homo_dim=0)
                pd_gt, bcp_gt, dcp_gt = compute_persistence_2DImg_1DHom_gt(-groundtruth_2*mm, padwidth = padwidth, homo_dim=0)
                bcp_lh *= 2
                dcp_lh *= 2
                bcp_gt *= 2
                dcp_gt *= 2
            else:
                pd_lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom_lh(-likelihood*mm, padwidth = padwidth, homo_dim=0)
                pd_gt, bcp_gt, dcp_gt = compute_persistence_2DImg_1DHom_gt(-groundtruth*mm, padwidth = padwidth, homo_dim=0)
            pers_thd_lh = 0.1
            #print('pd_lh.shape[0]',pd_lh.shape[0])
            if(pd_lh.shape[0] > 0):
                lh_pers = pd_lh[:, 1] - pd_lh[:, 0]
                lh_pers_valid = lh_pers[np.where(lh_pers > pers_thd_lh)];
            else:
                lh_pers =np.array([])
                lh_pers_valid = np.array([])

            pers_thd_gt = 0.0
            if(pd_gt.shape[0] > 0):
                gt_pers = pd_gt[:, 1] - pd_gt[:, 0]
                gt_pers_valid = gt_pers[np.where(gt_pers > pers_thd_gt)];
            else:
                gt_pers = np.array([])
                gt_pers_valid = np.array([]);

            using_lh_cp = True; 
            if(pd_lh.shape[0] > gt_pers_valid.shape[0]): 
                force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect = compute_dgm_force(pd_lh, pd_gt, pers_thresh=pers_thd_lh,pers_thresh_perfect=0.99, do_return_perfect=True)
                n_fix += len(idx_holes_to_fix);
                n_remove += len(idx_holes_to_remove);
                # print('len(idx_holes_to_fix)', len(idx_holes_to_fix))
                # print('len(idx_holes_to_remove)', len(idx_holes_to_remove))
                # print('len(idx_holes_perfect)', len(idx_holes_perfect))
                if(len(idx_holes_to_fix)>0 or len(idx_holes_to_remove ) > 0):
                    for h in range(min(1000,len(idx_holes_perfect))):
                        hole_indx = idx_holes_perfect[h];
                        if(int(bcp_lh[hole_indx][0]) >= sub_patch_border_width and int(bcp_lh[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(bcp_lh[hole_indx][1]) >= sub_patch_border_width and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]-sub_patch_border_width):
                            topo_cp_weight_map_vis_b_perf[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                        if(int(dcp_lh[hole_indx][0]) >= sub_patch_border_width and int(dcp_lh[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(dcp_lh[hole_indx][1]) >= sub_patch_border_width and int(dcp_lh[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                            topo_cp_weight_map_vis_d_perf[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood

                    for h in range(min(1000,len(idx_holes_to_fix))):
                        hole_indx = idx_holes_to_fix[h];
                        if(int(bcp_lh[hole_indx][0]) >= sub_patch_border_width and int(bcp_lh[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(bcp_lh[hole_indx][1]) >= sub_patch_border_width and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]-sub_patch_border_width):
                            topo_cp_weight_map[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                            topo_cp_weight_map_vis_b_fix[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                            topo_cp_ref_map[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1;
                        if(int(dcp_lh[hole_indx][0]) >= sub_patch_border_width and int(dcp_lh[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(dcp_lh[hole_indx][1]) >= sub_patch_border_width and int(dcp_lh[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                            topo_cp_weight_map[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                            topo_cp_weight_map_vis_d_fix[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                            topo_cp_ref_map[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 0; 

                    for h in range(min(1000,len(idx_holes_to_remove))):
                        hole_indx = idx_holes_to_remove[h];
                        if(int(bcp_lh[hole_indx][0]) >= sub_patch_border_width and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] - sub_patch_border_width and int(bcp_lh[hole_indx][1]) >= sub_patch_border_width and int(bcp_lh[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                            topo_cp_weight_map[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1; # push birth to death  # push to diagonal
                            topo_cp_weight_map_vis_b_rem[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1; # push birth to death  # push to diagonal
                            if(int(dcp_lh[hole_indx][0]) >= sub_patch_border_width and int(dcp_lh[hole_indx][0]) < likelihood.shape[0]- sub_patch_border_width and int(dcp_lh[hole_indx][1]) >= sub_patch_border_width and int(dcp_lh[hole_indx][1]) < likelihood.shape[1]- sub_patch_border_width):
                                topo_cp_ref_map[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = likelihood[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]; 
                            else:
                                topo_cp_ref_map[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = groundtruth[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])];  
                        if(int(dcp_lh[hole_indx][0]) >= sub_patch_border_width and int(dcp_lh[hole_indx][0]) < likelihood.shape[0] - sub_patch_border_width and int(dcp_lh[hole_indx][1]) >= sub_patch_border_width and int(dcp_lh[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                            topo_cp_weight_map[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 1; # push death to birth # push to diagonal
                            topo_cp_weight_map_vis_d_rem[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 1; # push death to birth # push to diagonal
                            if(int(bcp_lh[hole_indx][0]) >= sub_patch_border_width and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] - sub_patch_border_width and int(bcp_lh[hole_indx][1]) >= sub_patch_border_width and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]- sub_patch_border_width):
                                topo_cp_ref_map[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = likelihood[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]; 
                            else:
                                topo_cp_ref_map[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = groundtruth[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]; 
                if(len(idx_holes_to_fix) + len(idx_holes_perfect) < gt_pers_valid.shape[0]):
                    for hole_indx in range(gt_pers.shape[0]):
                        if(int(bcp_gt[hole_indx][0]) >= sub_patch_border_width and int(bcp_gt[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(bcp_gt[hole_indx][1]) >= sub_patch_border_width and int(bcp_gt[hole_indx][1]) < likelihood.shape[1]-sub_patch_border_width):
                            topo_cp_weight_map[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                            topo_cp_weight_map_vis_b_gt[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                            topo_cp_ref_map[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = groundtruth[int(bcp_gt[hole_indx][0]), int(bcp_gt[hole_indx][1])]; 
                        if(int(dcp_gt[hole_indx][0]) >= sub_patch_border_width and int(dcp_gt[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(dcp_gt[hole_indx][1]) >= sub_patch_border_width and int(dcp_gt[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                            topo_cp_weight_map[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                            topo_cp_weight_map_vis_d_gt[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                            topo_cp_ref_map[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = groundtruth[int(dcp_gt[hole_indx][0]), int(dcp_gt[hole_indx][1])]; 

            else:
                using_lh_cp = False;
                for hole_indx in range(gt_pers.shape[0]):
                    if(int(bcp_gt[hole_indx][0]) >= sub_patch_border_width and int(bcp_gt[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(bcp_gt[hole_indx][1]) >= sub_patch_border_width and int(bcp_gt[hole_indx][1]) < likelihood.shape[1]-sub_patch_border_width):
                        topo_cp_weight_map[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_weight_map_vis_b_gt[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = groundtruth[int(bcp_gt[hole_indx][0]), int(bcp_gt[hole_indx][1])]; 
                    if(int(dcp_gt[hole_indx][0]) >= sub_patch_border_width and int(dcp_gt[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(dcp_gt[hole_indx][1]) >= sub_patch_border_width and int(dcp_gt[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                        topo_cp_weight_map[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                        topo_cp_weight_map_vis_d_gt[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                        topo_cp_ref_map[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = groundtruth[int(dcp_gt[hole_indx][0]), int(dcp_gt[hole_indx][1])]; 
    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).to(device)
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).to(device)

    #print('topo_cp_ref_map.sum()',topo_cp_ref_map.sum())
    intersection = (et_sig * topo_cp_ref_map*topo_cp_weight_map).sum()
    union = ((et_sig*topo_cp_weight_map.squeeze(dim=1))**2).sum() + ((topo_cp_ref_map)**2).sum()
    loss_pers =  1 - ((2 * intersection + 1) / (union + 1))
    return loss_pers