import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """Alignment network to ensure rotation invariance."""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Initialize as identity matrix
        iden = torch.eye(self.k, requires_grad=True).repeat(batchsize, 1, 1)
        if x.is_cuda: iden = iden.cuda()
        x = x.view(-1, self.k, self.k) + iden
        return x

class PointNetSeg(nn.Module):
    """Lightweight PointNet for Semantic Segmentation."""
    def __init__(self, num_classes=9):
        super(PointNetSeg, self).__init__()
        self.stn = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.conv4 = nn.Conv1d(64, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

        # Segmentation head
        self.seg_conv1 = nn.Conv1d(2048 + 64, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, 128, 1)
        self.seg_conv4 = nn.Conv1d(128, num_classes, 1)
        
        self.s_bn1 = nn.BatchNorm1d(512)
        self.s_bn2 = nn.BatchNorm1d(256)
        self.s_bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        n_pts = x.size()[2]
        
        # Input Transform
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        # Feature Extraction
        x = F.relu(self.bn1(self.conv1(x)))
        point_features = x # Save for skip-connection (local features)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        
        # Global Feature (Max Pooling)
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature_repeated = global_feature.repeat(1, 1, n_pts)
        
        # Concatenate Global + Local Features
        x = torch.cat([point_features, global_feature_repeated], 1)
        
        # Segmentation Head
        x = F.relu(self.s_bn1(self.seg_conv1(x)))
        x = F.relu(self.s_bn2(self.seg_conv2(x)))
        x = F.relu(self.s_bn3(self.seg_conv3(x)))
        x = self.seg_conv4(x)
        
        return x.transpose(2, 1) # Return (B, N, num_classes)