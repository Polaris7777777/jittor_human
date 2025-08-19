import re
import matplotlib.pyplot as plt
import numpy as np

def parse_skeleton_training_log(log_file_path):
    """
    解析骨骼训练日志文件，提取各种损失数据
    """
    # 存储训练数据
    train_epochs = []
    train_losses = []
    
    # 存储验证数据
    val_epochs = []
    val_losses = []
    j2j_losses = []
    
    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    current_epoch = None
    
    for line in lines:
        line = line.strip()
        
        # 匹配训练损失行（包含Train Loss的行）
        train_match = re.search(r'Epoch \[(\d+)/\d+\] Train Loss: ([\d.]+)', line)
        if train_match:
            epoch = int(train_match.group(1))
            train_loss = float(train_match.group(2))
            
            train_epochs.append(epoch)
            train_losses.append(train_loss)
            current_epoch = epoch
        
        # 匹配验证损失行
        val_match = re.search(r'Validation Loss: ([\d.]+) J2J Loss: ([\d.]+)', line)
        if val_match and current_epoch is not None:
            val_loss = float(val_match.group(1))
            j2j_loss = float(val_match.group(2))
            
            val_epochs.append(current_epoch)
            val_losses.append(val_loss)
            j2j_losses.append(j2j_loss)
    
    return {
        'train': {
            'epochs': train_epochs,
            'losses': train_losses
        },
        'val': {
            'epochs': val_epochs,
            'val_losses': val_losses,
            'j2j_losses': j2j_losses
        }
    }

def plot_skeleton_losses(data, save_path=None):
    """
    绘制骨骼训练的各种损失趋势
    """
    # 设置matplotlib参数
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 12
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Skeleton Training Loss Trends', fontsize=16, fontweight='bold')
    
    # 1. 训练损失趋势
    axes[0, 0].plot(data['train']['epochs'], data['train']['losses'], 
                    'b-', linewidth=2, label='Train Loss', alpha=0.8, marker='o', markersize=3)
    axes[0, 0].set_title('Training Loss Trend', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Train Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')  # 使用对数坐标
    
    # 2. 验证损失趋势
    axes[0, 1].plot(data['val']['epochs'], data['val']['val_losses'], 
                    'r-', linewidth=2, label='Validation Loss', alpha=0.8, marker='s', markersize=3)
    axes[0, 1].set_title('Validation Loss Trend', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')  # 使用对数坐标
    
    # 3. J2J损失趋势
    axes[1, 0].plot(data['val']['epochs'], data['val']['j2j_losses'], 
                    'g-', linewidth=2, label='J2J Loss', alpha=0.8, marker='^', markersize=3)
    axes[1, 0].set_title('J2J Loss Trend', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('J2J Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')  # 使用对数坐标
    
    # 4. 所有损失对比
    axes[1, 1].plot(data['train']['epochs'], data['train']['losses'], 
                    'b-', linewidth=2, label='Train Loss', alpha=0.8)
    axes[1, 1].plot(data['val']['epochs'], data['val']['val_losses'], 
                    'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    axes[1, 1].plot(data['val']['epochs'], data['val']['j2j_losses'], 
                    'g-', linewidth=2, label='J2J Loss', alpha=0.8)
    axes[1, 1].set_title('All Losses Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')  # 使用对数坐标
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss trends plot saved to: {save_path}")
    
    plt.show()

def plot_losses_linear_scale(data, save_path=None):
    """
    使用线性坐标绘制损失趋势（更清楚地看到细节变化）
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Skeleton Training Loss Trends (Linear Scale)', fontsize=16, fontweight='bold')
    
    # 训练损失
    axes[0].plot(data['train']['epochs'], data['train']['losses'], 
                'b-', linewidth=2, label='Train Loss', alpha=0.8, marker='o', markersize=2)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 验证损失
    axes[1].plot(data['val']['epochs'], data['val']['val_losses'], 
                'r-', linewidth=2, label='Validation Loss', alpha=0.8, marker='s', markersize=2)
    axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # J2J损失
    axes[2].plot(data['val']['epochs'], data['val']['j2j_losses'], 
                'g-', linewidth=2, label='J2J Loss', alpha=0.8, marker='^', markersize=2)
    axes[2].set_title('J2J Loss', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Linear scale loss trends plot saved to: {save_path}")
    
    plt.show()

def print_loss_statistics(data):
    """
    打印损失统计信息
    """
    print("=" * 60)
    print("Skeleton Training Loss Statistics")
    print("=" * 60)
    
    # 训练损失统计
    train_loss_min = min(data['train']['losses'])
    train_loss_max = max(data['train']['losses'])
    train_loss_avg = np.mean(data['train']['losses'])
    train_loss_final = data['train']['losses'][-1]
    
    print("Training Loss Statistics:")
    print(f"  Min: {train_loss_min:.6f}, Max: {train_loss_max:.6f}")
    print(f"  Avg: {train_loss_avg:.6f}, Final: {train_loss_final:.6f}")
    print(f"  Improvement: {train_loss_max - train_loss_final:.6f} ({((train_loss_max - train_loss_final)/train_loss_max*100):.1f}%)")
    
    # 验证损失统计
    val_loss_min = min(data['val']['val_losses'])
    val_loss_max = max(data['val']['val_losses'])
    val_loss_avg = np.mean(data['val']['val_losses'])
    val_loss_final = data['val']['val_losses'][-1]
    
    print("\nValidation Loss Statistics:")
    print(f"  Min: {val_loss_min:.6f}, Max: {val_loss_max:.6f}")
    print(f"  Avg: {val_loss_avg:.6f}, Final: {val_loss_final:.6f}")
    print(f"  Improvement: {val_loss_max - val_loss_final:.6f} ({((val_loss_max - val_loss_final)/val_loss_max*100):.1f}%)")
    
    # J2J损失统计
    j2j_loss_min = min(data['val']['j2j_losses'])
    j2j_loss_max = max(data['val']['j2j_losses'])
    j2j_loss_avg = np.mean(data['val']['j2j_losses'])
    j2j_loss_final = data['val']['j2j_losses'][-1]
    
    print("\nJ2J Loss Statistics:")
    print(f"  Min: {j2j_loss_min:.6f}, Max: {j2j_loss_max:.6f}")
    print(f"  Avg: {j2j_loss_avg:.6f}, Final: {j2j_loss_final:.6f}")
    print(f"  Improvement: {j2j_loss_max - j2j_loss_final:.6f} ({((j2j_loss_max - j2j_loss_final)/j2j_loss_max*100):.1f}%)")
    
    # 找到最佳epoch
    best_val_epoch = data['val']['epochs'][data['val']['val_losses'].index(val_loss_min)]
    best_j2j_epoch = data['val']['epochs'][data['val']['j2j_losses'].index(j2j_loss_min)]
    
    print(f"\nBest Results:")
    print(f"  Best Validation Loss: {val_loss_min:.6f} at Epoch {best_val_epoch}")
    print(f"  Best J2J Loss: {j2j_loss_min:.6f} at Epoch {best_j2j_epoch}")

def main(log_file_path, output_path_dir):
    # 替换为您的日志文件路径
    # log_file_path = "training_log.txt"  # 请修改为实际的文件路径
    
    try:
        # 解析日志文件
        print("Parsing skeleton training log...")
        data = parse_skeleton_training_log(log_file_path)
        
        print(f"Successfully parsed {len(data['train']['epochs'])} training epochs")
        print(f"Successfully parsed {len(data['val']['epochs'])} validation epochs")
        
        # 打印统计信息
        print_loss_statistics(data)
        
        # 绘制损失趋势（对数坐标）
        print("\nPlotting loss trends (log scale)...")
        plot_skeleton_losses(data, save_path=f"{output_path_dir}/skeleton_losses_log_scale.png")
        
        # 绘制损失趋势（线性坐标）
        print("Plotting loss trends (linear scale)...")
        plot_losses_linear_scale(data, save_path=f"{output_path_dir}/skeleton_losses_linear_scale.png")
        
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found")
        print("Please ensure the file path is correct")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    log_file_path = 'output/sal_512_2048/skeleton/training_log.txt'
    output_path_dir = 'output/sal_512_2048/skeleton'
    main(log_file_path, output_path_dir)