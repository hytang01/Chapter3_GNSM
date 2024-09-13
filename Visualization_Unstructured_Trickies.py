import numpy as np


def unstructure_2_structure_trick(num_reals,num_ts,Nc, x_struct,y_struct,dx_struct,dy_struct, XYZ, saturation_pred, saturation_true, saturation_diff):
    # tasks:
        # converst state_var_true/pred/diff into state_var_true/repd/diff_struct
    # convert state_var
    state_var_struct = {}
    
        # first, calc the center coordinates of each cell in structured model
    center_coord = np.zeros((x_struct,y_struct,2))
    for i in range(x_struct):
        for j in range(y_struct):
            center_coord[i][j][0] = dx_struct/2 +dx_struct*i
            center_coord[i][j][1] = dy_struct/2 +dy_struct*j
#     print(center_coord[:,:,0]) 
        # second, loop over all cells in structured model and based on the center coord & XYZ from unstructured model to decide the closest cell in unstruct case
    state_var_struct['true'] = np.zeros((num_reals,num_ts,x_struct,y_struct))
    state_var_struct['pred'] =  np.zeros((num_reals,num_ts,x_struct,y_struct))
    state_var_struct['diff'] =  np.zeros((num_reals,num_ts,x_struct,y_struct))
    
    for i in range(x_struct):
        for j in range(y_struct):
            index = search_cloest_unstruct_cell(center_coord[i][j][0],center_coord[i][j][1],XYZ,Nc)
            state_var_struct['true'][:,:,i,j] = saturation_true[:,:,index,0]
            state_var_struct['pred'][:,:,i,j] = saturation_pred[:,:,index,0]
            state_var_struct['diff'][:,:,i,j] = saturation_diff[:,:,index,0]
                    
    return state_var_struct


def unstructure_2_structure_trick_single_well(Nc,dx_struct,dy_struct,well_idx, XYZ): # careful here well_idx is 1 based for adgprs input
    # tasks:
         # convert well to well_struct
    x_coord = XYZ[well_idx-1][0] # distance from the origion (0,0)
    y_coord = XYZ[well_idx-1][1]
    # now, calc the x/y cell idx in structured model
    well_x_struct = round((x_coord-dx_struct/2)/dx_struct) # 0-based
    well_y_struct = round((y_coord-dy_struct/2)/dy_struct) # 0-based
    return well_x_struct+0.5,well_y_struct+0.5 # make it center of the block in image


def search_cloest_unstruct_cell(center_coord_x, center_coord_y, XYZ, Nc):
    # x & y coord in structured model, XYZ & number of cells in unstructured model
    index = -1
    min_dist = 1e5
    for idx in range(Nc):
        dist = ((XYZ[idx][0]-center_coord_x)**2+(XYZ[idx][1]-center_coord_y)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            index = idx
    return index