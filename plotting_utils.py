#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘图和可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.figsize'] = (14, 10)
rcParams['font.size'] = 10


def plot_thermal_resistance_breakdown(R_dict, heat_flux, save_fig=True):
    """
    绘制热阻分布饼图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('热阻分析', fontsize=14, fontweight='bold')
    
    # 准备数据
    labels = []
    values = []
    for key, val in R_dict.items():
        if key != 'ambient_h_total' and isinstance(val, (int, float)):
            labels.append(key)
            values.append(val)
    
    # 1. 饼图
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax1.pie(values, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax1.set_title('热阻分布', fontsize=12, fontweight='bold')
    
    # 美化文字
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 2. 柱状图
    x_pos = np.arange(len(labels))
    ax2.bar(x_pos, values, color=colors, alpha=0.8)
    ax2.set_ylabel('热阻 (K/W)', fontsize=11, fontweight='bold')
    ax2.set_title('热阻柱状图', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(ax2.patches, values)):
        ax2.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('thermal_resistance.png', dpi=300, bbox_inches='tight')
        print("热阻分布图已保存为 'thermal_resistance.png'")
    
    plt.show()


if __name__ == "__main__":
    print("绘图工具模块")
