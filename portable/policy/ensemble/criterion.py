import torch


def L_metric(feat1, feat2, same_class=True):
    d = torch.sum((feat1 - feat2).pow(2).view((-1, feat1.size(-1))), 1)
    if same_class:
        return d.sum()/d.size(0)
    else:
        return torch.clamp(1 - d, min=0).sum() / d.size(0)


n_modules = None  # number of modules, global variable that needs to be initialized once
every_tuple = None  # initialized in criterion(), so that we don't repeatedly compute it in L_divergence()
def L_divergence(feats):
    """
    feats is of shape (n_modules, n_features)
    """
    every_tuple_features = feats[every_tuple, :]  # (num_tuple, 2, dim)
    every_tuple_difference = every_tuple_features.diff(dim=1).squeeze(1)  # (num_tuple, dim)
    loss = torch.clamp(1 - torch.sum(every_tuple_difference.pow(2), dim=-1), min=0)  # (num_tuple, )
    mean_loss = loss.mean(dim=0)
    return mean_loss


def batched_L_divergence(batch_feats):
    """
    batch_feats is of shape (batch_size, n_modules, n_features)
    """
    if batch_feats.shape[1] == 1:
        # if there is only one module, then there is no divergence
        return torch.Tensor([0]).to(batch_feats.device)

    global every_tuple
    if every_tuple is None:
        every_tuple = torch.combinations(torch.Tensor(range(batch_feats.shape[1])), 2).long()
        
    every_tuple_features = batch_feats[:, every_tuple, :]  # (batch_size, num_tuple, 2, dim)
    every_tuple_difference = every_tuple_features.diff(dim=2).squeeze(2)  # (batch_size, num_tuple, dim)
    loss = torch.clamp(1 - torch.mean(every_tuple_difference.pow(2), dim=-1), min=0)  # (batch_size, num_tuple)
    mean_loss = loss.mean()
    return mean_loss


def loss_function(tensor, batch_k):
    batch_size = tensor.size(0)
    assert batch_size % batch_k == 0
    assert batch_k > 1
    loss_homo, loss_heter, loss_div = 0, 0, 0
    for i in range(batch_size):
        loss_div += L_divergence(tensor[i, ...])
    count_homo, count_heter = 0, 0
    for group_index in range(batch_size // batch_k):
        for i in range(batch_k):
            anchor = tensor[i+group_index*batch_k: 1+i+group_index*batch_k, ...]
            for j in range(i + 1, batch_k):
                index = j + group_index*batch_k
                loss_homo += L_metric(anchor, tensor[index: 1 + index, ...])
                count_homo += 1
            for j in range((group_index + 1)*batch_k, batch_size):
                loss_heter += L_metric(anchor, tensor[j:j + 1, ...])
                count_heter += 1
    
    return loss_div/batch_size, loss_homo/count_homo, loss_heter/count_heter


def criterion(examples, class_label_matrix):
    # L_metric loss for examples of the same class
    class_label_matrix = torch.triu(class_label_matrix, diagonal=1)  # only consider the strict upper triangle
    positive_x, positive_y = torch.where(class_label_matrix == 1)
    loss_homo = L_metric(examples[positive_x, ...], examples[positive_y, ...], same_class=True)

    # L_metric loss for examples of different class
    class_label_matrix = class_label_matrix + torch.tril(torch.ones_like(class_label_matrix))  # change lower triangular of class_label to 1
    negative_x, negative_y = torch.where(class_label_matrix == 0)
    loss_heter = L_metric(examples[negative_x, ...], examples[negative_y, ...], False)

    # divergence loss
    global n_modules
    global every_tuple
    if n_modules is None:
        n_modules = examples.shape[1]  # init the global variable for L_divergence
        every_tuple = torch.combinations(torch.Tensor(range(n_modules)), 2).long()
    loss_div = batched_L_divergence(examples)
    
    return loss_div, loss_homo, loss_heter


def cluster_centroid_loss(cluster_a, cluster_b, margin=1):
    centroid_a = torch.mean(cluster_a, 0, keepdim=True)
    centroid_b = torch.mean(cluster_b, 0, keepdim=True)

    loss_a = torch.clamp(
        (cluster_a - centroid_a)**2 - (cluster_a - centroid_b)**2 + margin,
        min=0
    )

    loss_b = torch.clamp(
        (cluster_b - centroid_b)**2 - (cluster_b - centroid_a)**2 + margin,
        min=0
    )

    return loss_a + loss_b
