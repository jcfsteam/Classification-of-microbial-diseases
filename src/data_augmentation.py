# src/data_augmentation.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm
from src.models.diffusion_model import DiffusionModel, DiffusionTrainer, convert_to_tensor_dataset
from config.model_config import DiffusionConfig

class DataAugmentor:
    def __init__(self, config: DiffusionConfig):
        """
        初始化数据增强器
        
        Parameters:
        -----------
        config : DiffusionConfig
            扩散模型配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
            
    def augment_data(self, X_train: np.ndarray, y_train: np.ndarray,
                    augmentation_factor: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用扩散模型进行数据增强，确保各类别样本数量平衡，并保留所有原始样本
        
        Parameters:
        -----------
        X_train : np.ndarray
            训练数据特征
        y_train : np.ndarray
            训练数据标签
        augmentation_factor : float, optional
            增强因子，表示要生成的额外样本比例
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            增强后的数据集，包含所有原始样本，且每个类别具有相同数量的样本
        """
        if augmentation_factor is None:
            augmentation_factor = self.config.augmentation_factor
            
        print("\nInitializing data augmentation using diffusion model...")
        
        # 分析每个类别的样本数量
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        max_samples_per_class = max(label_counts)
        n_classes = len(unique_labels)
        
        # 确定每个类别的目标样本数（包括原始样本）
        target_samples_per_class = int(max_samples_per_class * (1 + augmentation_factor))
        
        # 准备数据
        dataset = convert_to_tensor_dataset(X_train, y_train, self.device)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # 初始化模型
        input_dim = X_train.shape[1]
        condition_dim = len(unique_labels)
        model = DiffusionModel(
            input_dim=input_dim,
            condition_dim=condition_dim,
            config=self.config
        ).to(self.device)
        
        # 初始化训练器
        trainer = DiffusionTrainer(config=self.config)
        
        # 训练扩散模型
        print("\nTraining diffusion model...")
        loss_history = trainer.train(
            model=model,
            train_dataloader=dataloader,
            num_epochs=self.config.num_epochs
        )
        
        print(f"\nTarget samples per class: {target_samples_per_class}")
        
        # 为每个类别准备数据
        final_samples = []
        final_labels = []
        
        for label, count in zip(unique_labels, label_counts):
            print(f"\nProcessing class {label} (current count: {count})")
            
            # 获取当前类别的原始样本
            mask = y_train == label
            X_class = X_train[mask]
            y_class = y_train[mask]
            
            # 计算需要生成的额外样本数
            samples_to_generate = target_samples_per_class - count
            print(f"Original samples: {count}")
            print(f"Additional samples to generate: {samples_to_generate}")
            
            if samples_to_generate > 0:
                # 生成新样本
                synthetic_samples = self._generate_samples(
                    model=model,
                    trainer=trainer,
                    num_samples=samples_to_generate,
                    label=label,
                    num_classes=n_classes
                )
                
                # 合并原始样本和生成的样本
                X_class_augmented = np.vstack([X_class, synthetic_samples])
                y_class_augmented = np.concatenate([y_class, np.array([label] * samples_to_generate)])
            else:
                # 如果这个类别已经有足够的样本，保留所有原始样本并复制一些样本到目标数量
                samples_to_duplicate = target_samples_per_class - count
                if samples_to_duplicate > 0:
                    # 随机选择一些原始样本进行复制
                    print(f"Duplicating {samples_to_duplicate} samples from class {label}")
                    indices = np.random.choice(len(X_class), samples_to_duplicate)
                    X_duplicate = X_class[indices]
                    y_duplicate = y_class[indices]
                    
                    # 合并原始样本和复制的样本
                    X_class_augmented = np.vstack([X_class, X_duplicate])
                    y_class_augmented = np.concatenate([y_class, y_duplicate])
                else:
                    X_class_augmented = X_class
                    y_class_augmented = y_class
            
            final_samples.append(X_class_augmented)
            final_labels.append(y_class_augmented)
            
            print(f"Final count for class {label}: {len(X_class_augmented)}")
        
        # 合并所有类别的样本
        X_augmented = np.vstack(final_samples)
        y_augmented = np.concatenate(final_labels)
        
        # 打印增强结果摘要
        print(f"\nData augmentation summary:")
        print(f"Original sample distribution:")
        for label, count in zip(unique_labels, label_counts):
            print(f"Class {label}: {count}")
        
        print(f"\nAugmented sample distribution:")
        unique_labels_aug, label_counts_aug = np.unique(y_augmented, return_counts=True)
        for label, count in zip(unique_labels_aug, label_counts_aug):
            print(f"Class {label}: {count}")
        
        print(f"\nOriginal total samples: {len(X_train)}")
        print(f"Augmented total samples: {len(X_augmented)}")
        
        # 验证所有原始样本都包含在增强后的数据集中
        original_samples_included = sum([np.any(np.all(X_augmented == x, axis=1)) 
                                    for x in X_train])
        print(f"Original samples included in augmented dataset: "
            f"{original_samples_included}/{len(X_train)}")
        
        return X_augmented, y_augmented
    
    def _generate_samples(self, model: DiffusionModel,
                         trainer: DiffusionTrainer,
                         num_samples: int,
                         label: int,
                         num_classes: int) -> np.ndarray:
        """
        为指定类别生成样本
        
        Parameters:
        -----------
        model : DiffusionModel
            训练好的扩散模型
        trainer : DiffusionTrainer
            扩散模型训练器
        num_samples : int
            要生成的样本数量
        label : int
            目标类别
        num_classes : int
            类别总数
            
        Returns:
        --------
        np.ndarray
            生成的样本
        """
        model.eval()
        with torch.no_grad():
            # 准备条件向量
            condition = torch.zeros(num_samples, num_classes).to(self.device)
            condition[:, label] = 1
            
            # 初始化随机噪声
            x = torch.randn(
                num_samples,
                model.net[0].in_features - model.time_embed[0].out_features - model.condition_embed[0].out_features
            ).to(self.device)
            
            # 反向扩散过程
            for t in tqdm(range(trainer.num_timesteps - 1, -1, -1),
                         desc="Generating samples"):
                t_batch = torch.ones(num_samples, dtype=torch.long,
                                   device=self.device) * t
                x = trainer.p_sample(model, x, t_batch, condition)
        
        return x.cpu().numpy()