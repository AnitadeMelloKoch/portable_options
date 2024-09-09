import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class PrintLayer(nn.Module):
    """
    A simple layer to print the shape of input tensors for debugging purposes.
    """
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

def generate_random_boxes(batch_size, num_boxes_per_image, image_height, image_width):
    """
    Generates random bounding boxes for each image in a batch.

    Args:
    - batch_size (int): Number of images in the batch.
    - num_boxes_per_image (int): Number of boxes to generate per image.
    - image_height (int): Height of the input images.
    - image_width (int): Width of the input images.

    Returns:
    - boxes (Tensor): A tensor where each row represents a box in the form
                      [batch_index, x1, y1, x2, y2].
    """
    boxes = []

    for i in range(batch_size):
        image_boxes = []
        for _ in range(num_boxes_per_image):
            # Generate random top-left corner (x1, y1)
            x1 = torch.randint(0, image_width - 5, (1,)).item()
            y1 = torch.randint(0, image_height - 5, (1,)).item()
            
            # Generate random bottom-right corner (x2, y2)
            x2 = torch.randint(x1 + 1, image_width, (1,)).item()
            y2 = torch.randint(y1 + 1, image_height, (1,)).item()
            
            # Append the box coordinates as a list
            image_boxes.append([i, float(x1), float(y1), float(x2), float(y2)])  # Convert to float

        # Concatenate boxes for this image into a tensor and append it
        boxes.append(torch.tensor(image_boxes, dtype=torch.float32))

    # Concatenate all image boxes into a single tensor
    return torch.cat(boxes, dim=0)

class RoI(nn.Module):
    """
    Region of Interest (RoI) pooling layer using `roi_align`.

    Args:
    - output_size (int or Tuple[int, int]): The size of the output feature map after pooling.
    - spatial_scale (float): Scale factor to map RoI coordinates to feature map size.
    """
    def __init__(self, output_size, spatial_scale):
        super(RoI, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, x):
        """
        Args:
        - x: Input feature maps from the backbone network, of shape (N, C, H, W)
        - rois: Tensor of shape (num_rois, 5), where each RoI is defined as (batch_index, x1, y1, x2, y2)

        Returns:
        - Pooled features of shape (num_rois, C, output_size[0], output_size[1])
        """
        # Generate random RoIs for the sake of demonstration
        rois = generate_random_boxes(64, 1, 8, 8)
        rois = rois.to(x.device).to(x.dtype)
        output = torchvision.ops.roi_align(x, rois, output_size=(self.output_size, self.output_size), spatial_scale=self.spatial_scale)
        return output

class RoI_CNN(nn.Module):
    """
    A CNN model with multiple heads, each using RoI pooling.

    Args:
    - num_classes (int): The number of classes for classification.
    - num_heads (int): The number of model heads to use.
    """
    def __init__(self, num_classes, num_heads):
        super(RoI_CNN, self).__init__()
        
        self.model = nn.ModuleList([nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Apply RoI pooling
            RoI(output_size=100, spatial_scale=1.0/16),
            
            nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Flatten(),           
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(num_classes)
        ) for _ in range(num_heads)])
        
        self.num_heads = num_heads
        self.num_classes = num_classes 

    def forward(self, x, logits=False):
        """
        Forward pass through the RoI_CNN model.

        Args:
        - x: Input tensor of shape (N, C, H, W).
        - logits (bool): If True, return raw logits, else return softmax probabilities.

        Returns:
        - pred: Tensor of shape (N, num_heads, num_classes) representing the predictions.
        """
        pred = torch.zeros(x.shape[0], self.num_heads, self.num_classes).to(x.device)
        
        for idx in range(self.num_heads):
            y = self.model[idx](x)
            if not logits:
                y = F.softmax(y, dim=-1)
            pred[:, idx, :] = y

        return pred
