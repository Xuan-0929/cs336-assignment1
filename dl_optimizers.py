#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习优化器模块 / Deep Learning Optimizers Module
实现交叉熵损失、梯度裁剪、SGD和AdamW优化器
Implements cross-entropy loss, gradient clipping, SGD and AdamW optimizers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
from collections.abc import Callable, Iterable
from typing import Optional
from torch import Tensor


# ==================== 交叉熵损失函数 / Cross-Entropy Loss ====================

def run_cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    计算交叉熵损失 / Compute cross-entropy loss
    
    实现了数值稳定的交叉熵损失函数，使用log-sum-exp技巧避免数值溢出
    Implements numerically stable cross-entropy loss using log-sum-exp trick to avoid overflow
    
    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): 未归一化的logits / Unnormalized logits
        targets (Int[Tensor, "batch_size"]): 目标类别索引 / Target class indices
    
    Returns:
        Float[Tensor, ""]: 平均交叉熵损失 / Average cross-entropy loss
    """
    # 数值稳定性处理：减去最大值 / Numerical stability: subtract maximum
    max_vals = torch.max(inputs, dim=1, keepdim=True)[0]
    shifted_logits = inputs - max_vals
    
    # 计算log softmax / Compute log softmax
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=1, keepdim=True))
    log_probs = shifted_logits - log_sum_exp
    
    # 获取目标token的log概率 / Get log probabilities of target tokens
    batch_size = inputs.shape[0]
    target_log_probs = log_probs[torch.arange(batch_size), targets]
    
    # 计算负对数似然损失 / Compute negative log-likelihood loss
    loss = -torch.mean(target_log_probs)
    
    return loss


def test_cross_entropy():
    """测试交叉熵损失函数的正确性和数值稳定性 / Test cross-entropy loss correctness and numerical stability"""
    print("=== 测试交叉熵损失函数 ===")
    print("=== Testing Cross-Entropy Loss ===")
    
    # 创建测试数据 / Create test data
    inputs = torch.tensor([
        [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
        [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
        [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
        [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
    ])
    targets = torch.tensor([1, 0, 2, 2])
    
    # 测试正常情况 / Test normal case
    expected = F.cross_entropy(inputs, targets)
    our_result = run_cross_entropy(inputs, targets)
    
    print(f"PyTorch官方结果: {expected.item():.6f}")
    print(f"PyTorch official result: {expected.item():.6f}")
    print(f"我们的实现结果: {our_result.item():.6f}")
    print(f"Our implementation result: {our_result.item():.6f}")
    print(f"误差: {abs(expected.item() - our_result.item()):.8f}")
    print(f"Error: {abs(expected.item() - our_result.item()):.8f}")
    
    # 数值稳定性测试 / Numerical stability test
    large_inputs = 1000.0 * inputs
    large_expected = F.cross_entropy(large_inputs, targets)
    large_our_result = run_cross_entropy(large_inputs, targets)
    
    print(f"\n数值稳定性测试 (Large inputs test):")
    print(f"PyTorch官方结果: {large_expected.item():.6f}")
    print(f"PyTorch official result: {large_expected.item():.6f}")
    print(f"我们的实现结果: {large_our_result.item():.6f}")
    print(f"Our implementation result: {large_our_result.item():.6f}")
    print(f"误差: {abs(large_expected.item() - large_our_result.item()):.8f}")
    print(f"Error: {abs(large_expected.item() - large_our_result.item()):.8f}")
    
    # 验证正确性 / Verify correctness
    numpy.testing.assert_allclose(
        our_result.detach().numpy(),
        expected.detach().numpy(),
        atol=1e-4,
    )
    
    numpy.testing.assert_allclose(
        large_our_result.detach().numpy(),
        large_expected.detach().numpy(),
        atol=1e-4,
    )
    
    print("✓ 交叉熵损失函数测试通过")
    print("✓ Cross-entropy loss test passed")


# ==================== 梯度裁剪 / Gradient Clipping ====================

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    梯度裁剪实现 / Gradient clipping implementation
    
    将参数梯度的L2范数裁剪到指定最大值以下
    Clip parameter gradients to have L2 norm at most max_l2_norm
    
    Args:
        parameters (Iterable[torch.nn.Parameter]): 可训练参数集合 / Trainable parameters
        max_l2_norm (float): 最大L2范数值 / Maximum L2 norm value
    
    The gradients of the parameters (parameter.grad) are modified in-place.
    """
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return
    
    # 计算所有梯度的L2范数 / Calculate L2 norm of all gradients
    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 计算裁剪系数 / Calculate clipping coefficient
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    
    # 如果总范数超过最大值，则裁剪梯度 / Clip gradients if total norm exceeds maximum
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.data.mul_(clip_coef)


def test_gradient_clipping():
    """测试梯度裁剪的正确性 / Test gradient clipping correctness"""
    print("\n=== 测试梯度裁剪 ===")
    print("\n=== Testing Gradient Clipping ===")
    
    # 创建测试参数 / Create test parameters
    tensors = [torch.randn((5, 5)) for _ in range(6)]
    max_norm = 1e-2
    
    # 第一组：使用PyTorch官方实现 / First group: use PyTorch official implementation
    t1 = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    t1[-1].requires_grad_(False)  # 测试冻结参数 / Test frozen parameter
    
    loss = torch.cat(t1).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(t1, max_norm)
    t1_grads = [torch.clone(t.grad) for t in t1 if t.grad is not None]
    
    # 第二组：使用我们的实现 / Second group: use our implementation
    t1_c = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    t1_c[-1].requires_grad_(False)
    
    loss_c = torch.cat(t1_c).sum()
    loss_c.backward()
    run_gradient_clipping(t1_c, max_norm)
    t1_c_grads = [torch.clone(t.grad) for t in t1_c if t.grad is not None]
    
    # 比较结果 / Compare results
    assert len(t1_grads) == len(t1_c_grads)
    
    for i, (t1_grad, t1_c_grad) in enumerate(zip(t1_grads, t1_c_grads)):
        numpy.testing.assert_allclose(
            t1_grad.detach().numpy(),
            t1_c_grad.detach().numpy(),
            atol=1e-6,
        )
    
    print("✓ 梯度裁剪测试通过")
    print("✓ Gradient clipping test passed")


# ==================== SGD优化器 / SGD Optimizer ====================

class SGD(torch.optim.Optimizer):
    """
    SGD优化器实现 / SGD Optimizer Implementation
    
    实现了带动量衰减的SGD优化器，学习率随迭代次数衰减
    Implements SGD optimizer with momentum decay, learning rate decays with iteration count
    """
    
    def __init__(self, params, lr=1e-3):
        """
        初始化SGD优化器 / Initialize SGD optimizer
        
        Args:
            params: 模型参数 / Model parameters
            lr: 学习率 / Learning rate
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """
        执行单步优化 / Perform single optimization step
        
        Args:
            closure: 闭包函数，用于重新计算损失 / Closure function to recompute loss
        
        Returns:
            loss: 损失值 / Loss value
        """
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率 / Get learning rate
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]  # 获取参数状态 / Get parameter state
                t = state.get("t", 0)  # 获取迭代次数 / Get iteration number
                grad = p.grad.data  # 获取梯度 / Get gradient
                
                # 更新参数：学习率随迭代次数衰减 / Update parameters: learning rate decays with iterations
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # 增加迭代次数 / Increment iteration number
        
        return loss


def demo_sgd():
    """演示SGD优化器 / Demonstrate SGD optimizer"""
    print("\n=== SGD优化器演示 ===")
    print("\n=== SGD Optimizer Demo ===")
    
    # 创建简单的线性模型 / Create simple linear model
    model = nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # 模拟数据 / Simulate data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    criterion = nn.MSELoss()
    
    print("训练几步观察学习率衰减效果...")
    print("Training several steps to observe learning rate decay effect...")
    
    for i in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"Step {i+1}: Loss = {loss.item():.6f}")
    
    print("✓ SGD优化器演示完成")
    print("✓ SGD optimizer demo completed")


# ==================== AdamW优化器 / AdamW Optimizer ====================

class AdamW(torch.optim.Optimizer):
    """
    AdamW优化器实现 / AdamW Optimizer Implementation
    
    Adam with weight decay decoupled from the learning rate
    实现了权重衰减与学习率解耦的Adam优化器
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, correct_bias=True):
        """
        初始化AdamW优化器 / Initialize AdamW optimizer
        
        Args:
            params: 模型参数 / Model parameters
            lr: 学习率 / Learning rate
            betas: 动量参数 / Momentum parameters
            eps: 数值稳定性参数 / Numerical stability parameter
            weight_decay: 权重衰减系数 / Weight decay coefficient
            correct_bias: 是否进行偏差修正 / Whether to correct bias
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """执行单步优化 / Perform single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            correct_bias = group['correct_bias']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # 初始化状态 / Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                state['step'] += 1
                
                # 更新一阶和二阶矩估计 / Update first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差修正 / Bias correction
                if correct_bias:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = lr
                
                # 权重衰减 / Weight decay
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                # 参数更新 / Parameter update
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


def demo_adamw():
    """演示AdamW优化器 / Demonstrate AdamW optimizer"""
    print("\n=== AdamW优化器演示 ===")
    print("\n=== AdamW Optimizer Demo ===")
    
    # 创建简单的线性模型 / Create simple linear model
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # 模拟数据 / Simulate data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    criterion = nn.MSELoss()
    
    print("训练几步观察AdamW效果...")
    print("Training several steps to observe AdamW effect...")
    
    for i in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"Step {i+1}: Loss = {loss.item():.6f}")
    
    print("✓ AdamW优化器演示完成")
    print("✓ AdamW optimizer demo completed")


def test_all_optimizers():
    """测试所有优化器 / Test all optimizers"""
    print("=== 测试所有深度学习优化器 ===")
    print("=== Testing All Deep Learning Optimizers ===")
    
    # 测试交叉熵损失 / Test cross-entropy loss
    test_cross_entropy()
    
    # 测试梯度裁剪 / Test gradient clipping
    test_gradient_clipping()
    
    # 演示SGD优化器 / Demo SGD optimizer
    demo_sgd()
    
    # 演示AdamW优化器 / Demo AdamW optimizer
    demo_adamw()
    
    print("\n✓ 所有深度学习优化器测试完成")
    print("\n✓ All deep learning optimizers tests completed")


if __name__ == "__main__":
    test_all_optimizers()