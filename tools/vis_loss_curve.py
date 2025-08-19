import re
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(log_file_path):
    """
    解析训练日志文件，提取训练和验证损失
    """
    train_epochs = []
    train_mse_losses = []
    train_l1_losses = []
    
    val_epochs = []
    val_mse_losses = []
    val_l1_losses = []
    
    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    current_epoch = None
    
    for line in lines:
        line = line.strip()
        
        # 匹配训练损失行（包含Train Loss的行）
        train_match = re.search(r'Epoch \[(\d+)/\d+\].*Train Loss mse: ([\d.]+) Train Loss l1: ([\d.]+)', line)
        if train_match:
            epoch = int(train_match.group(1))
            mse_loss = float(train_match.group(2))
            l1_loss = float(train_match.group(3))
            
            train_epochs.append(epoch)
            train_mse_losses.append(mse_loss)
            train_l1_losses.append(l1_loss)
            current_epoch = epoch
        
        # 匹配验证损失行
        val_match = re.search(r'Validation Loss: mse: ([\d.]+) l1: ([\d.]+)', line)
        if val_match and current_epoch is not None:
            mse_loss = float(val_match.group(1))
            l1_loss = float(val_match.group(2))
            
            val_epochs.append(current_epoch)
            val_mse_losses.append(mse_loss)
            val_l1_losses.append(l1_loss)
    
    return {
        'train': {
            'epochs': train_epochs,
            'mse_losses': train_mse_losses,
            'l1_losses': train_l1_losses
        },
        'val': {
            'epochs': val_epochs,
            'mse_losses': val_mse_losses,
            'l1_losses': val_l1_losses
        }
    }

def plot_losses_separately(data, save_path=None):
    """
    分开绘制训练和验证损失
    """
    # 设置中文字体
    # plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
    # plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Train vs Val', fontsize=16, fontweight='bold')
    
    # 训练MSE损失
    axes[0, 0].plot(data['train']['epochs'], data['train']['mse_losses'], 
                    'b-', linewidth=2, label='Train MSE Loss', alpha=0.8)
    axes[0, 0].set_title('Train MSE Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 验证MSE损失
    axes[0, 1].plot(data['val']['epochs'], data['val']['mse_losses'], 
                    'r-', linewidth=2, label='Val MSE Loss', alpha=0.8)
    axes[0, 1].set_title('Val MSE Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 训练L1损失
    axes[1, 0].plot(data['train']['epochs'], data['train']['l1_losses'], 
                    'g-', linewidth=2, label='Train L1 Loss', alpha=0.8)
    axes[1, 0].set_title('Train L1 Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L1 Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 验证L1损失
    axes[1, 1].plot(data['val']['epochs'], data['val']['l1_losses'], 
                    'orange', linewidth=2, label='Val L1 Loss', alpha=0.8)
    axes[1, 1].set_title('Val L1 Loss', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('L1 Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def plot_losses_comparison(data, save_path=None):
    """
    在同一图中对比训练和验证损失
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('train vs val', fontsize=16, fontweight='bold')

    # MSE损失对比
    ax1.plot(data['train']['epochs'], data['train']['mse_losses'], 
             'b-', linewidth=2, label='Train MSE', alpha=0.8)
    ax1.plot(data['val']['epochs'], data['val']['mse_losses'], 
             'r-', linewidth=2, label='Val MSE', alpha=0.8)
    ax1.set_title('MSE Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # L1损失对比
    ax2.plot(data['train']['epochs'], data['train']['l1_losses'], 
             'g-', linewidth=2, label='Train L1', alpha=0.8)
    ax2.plot(data['val']['epochs'], data['val']['l1_losses'], 
             'orange', linewidth=2, label='Val L1', alpha=0.8)
    ax2.set_title('L1 Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L1 Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图表已保存到: {save_path}")
    
    plt.show()

def print_statistics(data):
    """
    打印损失统计信息
    """
    print("=" * 50)
    print("训练与验证损失统计")
    print("=" * 50)
    
    # 训练损失统计
    train_mse_min = min(data['train']['mse_losses'])
    train_mse_max = max(data['train']['mse_losses'])
    train_mse_avg = np.mean(data['train']['mse_losses'])
    train_mse_final = data['train']['mse_losses'][-1]
    
    train_l1_min = min(data['train']['l1_losses'])
    train_l1_max = max(data['train']['l1_losses'])
    train_l1_avg = np.mean(data['train']['l1_losses'])
    train_l1_final = data['train']['l1_losses'][-1]
    
    print("训练损失统计:")
    print(f"  MSE - 最小值: {train_mse_min:.6f}, 最大值: {train_mse_max:.6f}")
    print(f"        平均值: {train_mse_avg:.6f}, 最终值: {train_mse_final:.6f}")
    print(f"  L1  - 最小值: {train_l1_min:.6f}, 最大值: {train_l1_max:.6f}")
    print(f"        平均值: {train_l1_avg:.6f}, 最终值: {train_l1_final:.6f}")
    
    # 验证损失统计
    val_mse_min = min(data['val']['mse_losses'])
    val_mse_max = max(data['val']['mse_losses'])
    val_mse_avg = np.mean(data['val']['mse_losses'])
    val_mse_final = data['val']['mse_losses'][-1]
    
    val_l1_min = min(data['val']['l1_losses'])
    val_l1_max = max(data['val']['l1_losses'])
    val_l1_avg = np.mean(data['val']['l1_losses'])
    val_l1_final = data['val']['l1_losses'][-1]
    
    print("\n验证损失统计:")
    print(f"  MSE - 最小值: {val_mse_min:.6f}, 最大值: {val_mse_max:.6f}")
    print(f"        平均值: {val_mse_avg:.6f}, 最终值: {val_mse_final:.6f}")
    print(f"  L1  - 最小值: {val_l1_min:.6f}, 最大值: {val_l1_max:.6f}")
    print(f"        平均值: {val_l1_avg:.6f}, 最终值: {val_l1_final:.6f}")
    
    # 训练验证差异
    print(f"\n最终损失差异:")
    print(f"  MSE差异: {abs(train_mse_final - val_mse_final):.6f}")
    print(f"  L1差异: {abs(train_l1_final - val_l1_final):.6f}")

# 主函数
def main(log_file_path, output_dir='output_pct/skin'):   
    try:
        # 解析日志文件
        print("正在解析训练日志...")
        data = parse_training_log(log_file_path)
        
        print(f"成功解析 {len(data['train']['epochs'])} 个训练epoch")
        print(f"成功解析 {len(data['val']['epochs'])} 个验证epoch")
        
        # 打印统计信息
        print_statistics(data)
        
        # 分开绘制损失
        print("\n绘制分离的损失图表...")
        plot_losses_separately(data, save_path=f"{output_dir}/losses_separate.png")
        
        # 对比绘制损失
        print("绘制对比损失图表...")
        plot_losses_comparison(data, save_path=f"{output_dir}/losses_comparison.png")
        
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 '{log_file_path}'")
        print("请确保文件路径正确，或创建示例数据")
        
        # 创建示例数据进行演示
        create_sample_data(output_dir)
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

def create_sample_data(output_dir='output_pct/skin'):
    """
    基于您提供的日志格式创建示例数据
    """
    print("创建示例数据进行演示...")
    
    # 模拟数据
    epochs = list(range(880, 900))
    train_mse = [0.0046 + 0.001 * np.sin(i * 0.1) + np.random.normal(0, 0.0005) for i in range(len(epochs))]
    train_l1 = [0.0156 + 0.003 * np.sin(i * 0.1) + np.random.normal(0, 0.001) for i in range(len(epochs))]
    val_mse = [0.0045 + 0.001 * np.sin(i * 0.1) + np.random.normal(0, 0.0005) for i in range(len(epochs))]
    val_l1 = [0.0143 + 0.003 * np.sin(i * 0.1) + np.random.normal(0, 0.001) for i in range(len(epochs))]
    
    # 确保损失值为正
    train_mse = [max(0.001, x) for x in train_mse]
    train_l1 = [max(0.005, x) for x in train_l1]
    val_mse = [max(0.001, x) for x in val_mse]
    val_l1 = [max(0.005, x) for x in val_l1]
    
    sample_data = {
        'train': {
            'epochs': epochs,
            'mse_losses': train_mse,
            'l1_losses': train_l1
        },
        'val': {
            'epochs': epochs,
            'mse_losses': val_mse,
            'l1_losses': val_l1
        }
    }
    
    print_statistics(sample_data)
    plot_losses_separately(sample_data, save_path=f"{output_dir}/sample_losses_separate.png")
    plot_losses_comparison(sample_data, save_path=f"{output_dir}/sample_losses_comparison.png")

if __name__ == "__main__":
    log_file_path = 'output_pct/skin/training_log.txt'
    output_dir = '/'.join(log_file_path.split('/')[:-1])
    print(output_dir)
    main(log_file_path, output_dir=output_dir)