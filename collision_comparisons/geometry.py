import torch
import numpy as np

def box_sdf(points, box_center, box_size):
    '''Compute the signed distance function for boxes at the given points.

    Args:
        points (torch.Tensor): The points to compute the signed distance function for. Shape (*batch_shape, n_points, 3)
        box_center (torch.Tensor): The center of the box. Shape (n_boxes, 3)
        box_size (torch.Tensor): The size of the box. Shape (n_boxes, 3)
    
    Returns:
        torch.Tensor: The signed distance function for the box at the given points. Shape (*batch_shape, n_points, n_boxes)
    '''
    obs_pos = torch.as_tensor(box_center, dtype=points.dtype, device=points.device)
    obs_size = torch.as_tensor(box_size, dtype=points.dtype, device=points.device)
    p = (points[...,None,:] - obs_pos) #B x n_p x n_b x 3
    temp_q = torch.abs(p) - obs_size/2
    dist = torch.linalg.norm(torch.clamp(temp_q,min=0.0), dim=-1)
    dist2 = torch.max(temp_q,dim=-1)[0]
    mask = dist2<0
    dist[mask] = dist[mask] + dist2[mask]
    return dist


def points_sdf(other_points, point_pos):
    '''Compute the signed distance function for points at the given other points.

    Args:
        points (torch.Tensor): The points to compute the signed distance function for. Shape (*batch_shape, n_other_points, 3)
        point_pos (torch.Tensor): The position of the point. Shape (n_points, 3)
    
    Returns:
        torch.Tensor: The signed distance function for the point at the given points. Shape (*batch_shape, n_other_points, n_points)
    '''
    point_pos = torch.as_tensor(point_pos, dtype=other_points.dtype, device=other_points.device)
    return torch.linalg.norm(other_points[...,None,:] - point_pos, dim=-1)


def rot_mat(q, rot_axis):
    q = torch.as_tensor(q)
    # Get the skew symmetric matrix for the cross product
    e = rot_axis/torch.linalg.vector_norm(rot_axis)
    U = torch.tensor(
        [[0, -e[2], e[1]],
        [e[2], 0, -e[0]],
        [-e[1], e[0], 0]], dtype=q.dtype, device=q.device)
    # Compute for C and use broadcasting
    cq = torch.cos(q)[..., None, None]
    sq = torch.sin(q)[..., None, None]
    return torch.eye(3, dtype=q.dtype, device=q.device) + sq*U + (1-cq)*U@U