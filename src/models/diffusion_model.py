# src/models/diffusion_model.py

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from config.model_config import DiffusionConfig

class DiffusionModel(nn.Module):
    """
    条件扩散模型的神经网络实现
    """
    def __init__(self, input_dim: int, condition_dim: int,
                 config: DiffusionConfig):
        """
        初始化扩散模型
        
        Parameters:
        -----------
        input_dim : int
            输入特征维度
        condition_dim : int
            条件特征维度
        config : DiffusionConfig
            模型配置
        """
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        
        # 时间嵌入层
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 条件嵌入层
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 主要网络结构
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Parameters:
        -----------
        x : torch.Tensor
            输入数据
        t : torch.Tensor
            时间步
        condition : torch.Tensor
            条件信息

        Returns:
        --------
        torch.Tensor
            模型输出
        """
        t_emb = self.time_embed(t.unsqueeze(-1))
        c_emb = self.condition_embed(condition)
        h = torch.cat([x, t_emb, c_emb], dim=-1)
        return self.net(h)

class DiffusionTrainer:
    """
    扩散模型训练器
    """
    def __init__(self, config: DiffusionConfig):
        """
        初始化训练器
        
        Parameters:
        -----------
        config : DiffusionConfig
            训练配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_timesteps = config.num_timesteps
        
        # 设置噪声调度
        self.beta = torch.linspace(
            config.beta_start,
            config.beta_end,
            config.num_timesteps
        ).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        self._prepare_noise_schedule()
        
    def _prepare_noise_schedule(self):
        """准备噪声调度相关参数"""
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)
        
    def q_sample(self, x_0: torch.Tensor,
                 t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从扩散过程中采样
        
        Parameters:
        -----------
        x_0 : torch.Tensor
            原始数据
        t : torch.Tensor
            时间步
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            添加噪声后的数据和噪声
        """
        noise = torch.randn_like(x_0).to(self.device)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].reshape(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1)
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise
    
    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor,
                 t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        从噪声中采样生成数据
        
        Parameters:
        -----------
        model : nn.Module
            训练好的模型
        x_t : torch.Tensor
            当前时间步的数据
        t : torch.Tensor
            时间步
        condition : torch.Tensor
            条件信息
            
        Returns:
        --------
        torch.Tensor
            生成的样本
        """
        betas_t = self.beta[t].reshape(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1)
        sqrt_alpha_t = torch.sqrt(self.alpha[t]).reshape(-1, 1)
        
        model_mean = 1 / sqrt_alpha_t * (
            x_t - (betas_t * model(x_t, t/self.num_timesteps, condition)) / 
            sqrt_one_minus_alpha_bar_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(betas_t) * noise

    def train(self, model: nn.Module, train_dataloader: DataLoader,
              num_epochs: Optional[int] = None) -> list:
        """
        训练扩散模型
        
        Parameters:
        -----------
        model : nn.Module
            要训练的模型
        train_dataloader : DataLoader
            训练数据加载器
        num_epochs : int, optional
            训练轮数
            
        Returns:
        --------
        list
            训练损失历史
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
            
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate
        )
        
        loss_history = []
        model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_X, batch_y in pbar:
                    batch_size = batch_X.shape[0]
                    
                    # 采样随机时间步
                    t = torch.randint(
                        0, self.num_timesteps,
                        (batch_size,)
                    ).to(self.device)
                    
                    # 添加噪声
                    x_t, noise = self.q_sample(batch_X, t)
                    
                    # 预测噪声
                    predicted_noise = model(
                        x_t,
                        t/self.num_timesteps,
                        batch_y
                    )
                    
                    # 计算损失
                    loss = F.mse_loss(predicted_noise, noise)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_dataloader)
            loss_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
        
        return loss_history

def convert_to_tensor_dataset(X: np.ndarray, y: np.ndarray,
                            device: torch.device) -> TensorDataset:
    """
    将NumPy数组转换为PyTorch数据集
    
    Parameters:
    -----------
    X : np.ndarray
        特征数据
    y : np.ndarray
        标签数据
    device : torch.device
        设备类型
        
    Returns:
    --------
    TensorDataset
        PyTorch数据集
    """
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = F.one_hot(
        torch.tensor(y, dtype=torch.long),
        num_classes=len(np.unique(y))
    ).float().to(device)
    return TensorDataset(X_tensor, y_tensor)