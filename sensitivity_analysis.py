#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
敏感性分析
"""

import numpy as np
from thermal_calculator import ThermalMultilayerAnalyzer
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def sensitivity_analysis(analyzer, materials, T_gas, T_ambient, save_fig=True):
    """
    进行敏感性分析
    
    分析:
    1. 风速对外壳温度的影响
    2. 材料厚度对温度分布的影响
    3. 烟气温度对外壳温度的影响
    """
    print("\n" + "="*70)
    print("敏感性分析")
    print("="*70)
    
    # 1. 风速影响分析
    print("\n(1) 风速影响分析")
    print("-"*70)
    wind_speeds = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    outer_temps = []
    
    for ws in wind_speeds:
        analyzer_ws = ThermalMultilayerAnalyzer(
            materials_list=materials,
            T_gas=T_gas,
            T_ambient=T_ambient,
            wind_speed=ws,
            equipment_diameter=analyzer.equipment_diameter
        )
        temps, q = analyzer_ws.solve_steady_state()
        outer_temps.append(temps[-1] - 273.15)
        
        print(f"风速 {ws:>5.1f} m/s → 外壳温度 {temps[-1]-273.15:>7.2f}°C, 热流 {q:>10.2f} W/m²")
    
    # 2. 烟气温度影响分析
    print("\n(2) 烟气温度影响分析")
    print("-"*70)
    T_gas_values = np.array([800, 900, 1000, 1100, 1200, 1300]) + 273.15
    outer_temps_gas = []
    heat_fluxes_gas = []
    
    for Tg in T_gas_values:
        analyzer_tg = ThermalMultilayerAnalyzer(
            materials_list=materials,
            T_gas=Tg,
            T_ambient=T_ambient,
            wind_speed=analyzer.wind_speed,
            equipment_diameter=analyzer.equipment_diameter
        )
        temps, q = analyzer_tg.solve_steady_state()
        outer_temps_gas.append(temps[-1] - 273.15)
        heat_fluxes_gas.append(q)
        
        print(f"烟气温度 {Tg-273.15:>6.0f}°C → 外壳温度 {temps[-1]-273.15:>7.2f}°C, 热流 {q:>10.2f} W/m²")
    
    # 3. 环境温度影响分析
    print("\n(3) 环境温度影响分析")
    print("-"*70)
    T_amb_values = np.array([0, 10, 20, 30, 40, 50]) + 273.15
    outer_temps_amb = []
    
    for Ta in T_amb_values:
        analyzer_ta = ThermalMultilayerAnalyzer(
            materials_list=materials,
            T_gas=T_gas,
            T_ambient=Ta,
            wind_speed=analyzer.wind_speed,
            equipment_diameter=analyzer.equipment_diameter
        )
        temps, q = analyzer_ta.solve_steady_state()
        outer_temps_amb.append(temps[-1] - 273.15)
        
        print(f"环境温度 {Ta-273.15:>5.0f}°C → 外壳温度 {temps[-1]-273.15:>7.2f}°C")
    
    # 绘制敏感性分析结果
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('敏感性分析', fontsize=16, fontweight='bold')
    
    # 风速影响
    ax1 = axes[0, 0]
    ax1.plot(wind_speeds, outer_temps, 'o-', linewidth=2.5, markersize=8, color='#FF6B6B')
    ax1.set_xlabel('风速 (m/s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('外壳温度 (°C)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) 风速对外壳温度的影响', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    for x, y in zip(wind_speeds, outer_temps):
        ax1.text(x, y + 0.5, f'{y:.1f}', ha='center', fontsize=9)
    
    # 烟气温度影响
    ax2 = axes[0, 1]
    ax2.plot(T_gas_values - 273.15, outer_temps_gas, 'o-', linewidth=2.5, markersize=8, color='#4ECDC4')
    ax2.set_xlabel('烟气温度 (°C)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('外壳温度 (°C)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) 烟气温度对外壳温度的影响', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 热流随烟气温度变化
    ax3 = axes[1, 0]
    ax3.plot(T_gas_values - 273.15, heat_fluxes_gas, 's-', linewidth=2.5, markersize=8, color='#95E1D3')
    ax3.set_xlabel('烟气温度 (°C)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('热流密度 (W/m²)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) 热流随烟气温度的变化', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 环境温度影响
    ax4 = axes[1, 1]
    ax4.plot(T_amb_values - 273.15, outer_temps_amb, 'D-', linewidth=2.5, markersize=8, color='#FFB84D')
    ax4.set_xlabel('环境温度 (°C)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('外壳温度 (°C)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) 环境温度对外壳温度的影响', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        print("\n敏感性分析图已保存为 'sensitivity_analysis.png'")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("敏感性分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    print("敏感性分析模块")
