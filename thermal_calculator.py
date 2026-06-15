#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热工设备多层绝热结构温度计算程序

功能:
1. 计算多层耐火材料和保温材料的温度分布
2. 支持温度依赖的导热系数(二项式模型)
3. 考虑烟气侧对流换热
4. 考虑环境侧对流和辐射换热
5. 绘制温度-厚度曲线和详细的温度分析图表

作者: Thermal Engineering Team
日期: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import fsolve, minimize_scalar
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class ThermalMaterial:
    """
    耐火/保温材料类
    
    导热系数采用二项式温度依赖模型:
    k(T) = k0 * (1 + a*T + b*T^2)
    其中T为绝对温度(K)
    """
    
    def __init__(self, name, thickness, k0, a=0.0, b=0.0, density=1.0, specific_heat=1.0, emissivity=0.8):
        """
        初始化材料参数
        
        参数:
            name: 材料名称
            thickness: 材料厚度 [m]
            k0: 参考导热系数 [W/(m·K)] (在参考温度处, 通常为300K)
            a: 一次项系数
            b: 二次项系数
            density: 密度 [kg/m³]
            specific_heat: 比热容 [J/(kg·K)]
            emissivity: 发射率 [-]
        """
        self.name = name
        self.thickness = thickness
        self.k0 = k0
        self.a = a
        self.b = b
        self.density = density
        self.specific_heat = specific_heat
        self.emissivity = emissivity
    
    def thermal_conductivity(self, T):
        """
        计算在温度T处的导热系数
        
        参数:
            T: 温度 [K]
        
        返回值:
            导热系数 [W/(m·K)]
        """
        # 归一化温度(以300K为参考)
        T_norm = (T - 300.0) / 300.0
        k = self.k0 * (1.0 + self.a * T_norm + self.b * T_norm**2)
        return max(k, 0.01)  # 确保导热系数为正
    
    def thermal_conductivity_mean(self, T1, T2):
        """
        计算两个温度之间的平均导热系数
        
        参数:
            T1, T2: 温度 [K]
        
        返回值:
            平均导热系数 [W/(m·K)]
        """
        # 使用数值积分计算平均值
        def integrand(T):
            return self.thermal_conductivity(T)
        
        T_min, T_max = min(T1, T2), max(T1, T2)
        if abs(T_max - T_min) < 1:
            return self.thermal_conductivity((T_min + T_max) / 2)
        
        k_mean, _ = quad(integrand, T_min, T_max, limit=100)
        k_mean /= (T_max - T_min)
        return k_mean


class ConvectionModel:
    """
    对流换热模型
    """
    
    @staticmethod
    def churchill_bernstein_cylinder(Re, Pr):
        """
        Churchill-Bernstein关系式(圆柱形)
        用于计算横流圆柱的Nusselt数
        
        参数:
            Re: Reynolds数
            Pr: Prandtl数
        
        返回值:
            Nusselt数
        """
        if Re < 0.4:
            Nu = 0.891 * Re**0.33 * Pr**0.33
        elif Re < 40:
            Nu = 0.821 * Re**0.385 * Pr**0.33
        elif Re < 4000:
            Nu = 0.615 * Re**0.466 * Pr**0.33
        else:
            Nu = 0.174 * Re**0.618 * Pr**0.33
        return Nu
    
    @staticmethod
    def nusselt_to_h(Nu, k_air, D):
        """
        从Nusselt数计算对流换热系数
        h = Nu * k / D
        
        参数:
            Nu: Nusselt数
            k_air: 空气导热系数 [W/(m·K)]
            D: 特征长度(直径) [m]
        
        返回值:
            对流换热系数 [W/(m²·K)]
        """
        return Nu * k_air / D


class RadiationModel:
    """
    辐射换热模型
    """
    
    STEFAN_BOLTZMANN = 5.67e-8  # Stefan-Boltzmann常数 [W/(m²·K⁴)]
    
    @classmethod
    def radiation_heat_transfer(cls, T_surface, T_ambient, emissivity):
        """
        计算辐射换热热流
        Q = ε·σ·(T_surface⁴ - T_ambient⁴)
        
        参数:
            T_surface: 表面温度 [K]
            T_ambient: 环境温度 [K]
            emissivity: 发射率 [-]
        
        返回值:
            辐射热流 [W/m²]
        """
        q_rad = emissivity * cls.STEFAN_BOLTZMANN * (T_surface**4 - T_ambient**4)
        return q_rad
    
    @classmethod
    def radiation_heat_transfer_coefficient(cls, T_surface, T_ambient, emissivity):
        """
        计算辐射换热的等效线性化系数
        h_rad = ε·σ·(T_surface² + T_ambient²)·(T_surface + T_ambient)
        
        参数:
            T_surface: 表面温度 [K]
            T_ambient: 环境温度 [K]
            emissivity: 发射率 [-]
        
        返回值:
            等效辐射换热系数 [W/(m²·K)]
        """
        T_s2 = T_surface**2
        T_a2 = T_ambient**2
        h_rad = emissivity * cls.STEFAN_BOLTZMANN * (T_s2 + T_a2) * (T_surface + T_ambient)
        return h_rad


class AirProperties:
    """
    空气物性参数计算
    """
    
    @staticmethod
    def properties_at_temperature(T):
        """
        计算空气在温度T处的物性参数
        基于Sutherland公式和相关的气体运动论
        
        参数:
            T: 温度 [K]
        
        返回值:
            字典,包含:
            - rho: 密度 [kg/m³]
            - cp: 比热容 [J/(kg·K)]
            - k: 导热系数 [W/(m·K)]
            - mu: 动力粘度 [Pa·s]
            - Pr: Prandtl数
        """
        # 参考状态(300K, 101325 Pa)
        rho_ref = 1.177
        cp_ref = 1005.0
        k_ref = 0.0263
        mu_ref = 1.846e-5
        T_ref = 300.0
        
        # 理想气体密度变化
        rho = rho_ref * (T_ref / T)
        
        # 比热容随温度变化(多项式拟合)
        cp = 1000.0 + 0.1 * (T - T_ref) + 0.0001 * (T - T_ref)**2
        
        # Sutherland导热系数公式
        S_k = 194.0  # Sutherland常数
        k = k_ref * ((T / T_ref)**1.5) * (T_ref + S_k) / (T + S_k)
        
        # Sutherland动力粘度公式
        S_mu = 110.0  # Sutherland常数
        mu = mu_ref * ((T / T_ref)**1.5) * (T_ref + S_mu) / (T + S_mu)
        
        # Prandtl数
        Pr = (cp * mu) / k
        
        return {
            'rho': rho,
            'cp': cp,
            'k': k,
            'mu': mu,
            'Pr': Pr
        }


class ThermalMultilayerAnalyzer:
    """
    多层绝热结构热传导分析器
    """
    
    def __init__(self, materials_list, T_gas, T_ambient, h_gas=None, h_ambient=None, 
                 wind_speed=2.0, equipment_diameter=1.0):
        """
        初始化分析器
        
        参数:
            materials_list: 材料列表 [ThermalMaterial, ...]
            T_gas: 烟气温度 [K]
            T_ambient: 环境温度 [K]
            h_gas: 烟气侧对流系数 [W/(m²·K)], 如果为None则自动计算
            h_ambient: 环境侧对流系数 [W/(m²·K)], 如果为None则自动计算
            wind_speed: 风速 [m/s]
            equipment_diameter: 设备外径 [m]
        """
        self.materials = materials_list
        self.T_gas = T_gas
        self.T_ambient = T_ambient
        self.wind_speed = wind_speed
        self.equipment_diameter = equipment_diameter
        
        # 计算气体侧对流系数
        if h_gas is None:
            self.h_gas = self._calculate_gas_side_h()
        else:
            self.h_gas = h_gas
        
        # 计算环境侧对流系数
        if h_ambient is None:
            self.h_ambient = self._calculate_ambient_side_h()
        else:
            self.h_ambient = h_ambient
        
        print(f"\n=== 对流换热系数 ===")
        print(f"烟气侧对流系数: {self.h_gas:.2f} W/(m²·K)")
        print(f"环境侧对流系数: {self.h_ambient:.2f} W/(m²·K)")
    
    def _calculate_gas_side_h(self):
        """
        计算烟气侧对流系数
        假设高速烟气横流
        """
        # 烟气平均温度
        T_mean = (self.T_gas + self.T_ambient) / 2
        air_props = AirProperties.properties_at_temperature(T_mean)
        
        # 假设烟气流速
        v_gas = 10.0  # 高速热烟气,m/s
        
        # Reynolds数
        Re = (air_props['rho'] * v_gas * self.equipment_diameter) / air_props['mu']
        
        # Nusselt数
        Nu = ConvectionModel.churchill_bernstein_cylinder(Re, air_props['Pr'])
        
        # 对流系数
        h = ConvectionModel.nusselt_to_h(Nu, air_props['k'], self.equipment_diameter)
        
        return h
    
    def _calculate_ambient_side_h(self):
        """
        计算环境侧对流系数
        """
        # 环境温度下的空气物性
        air_props = AirProperties.properties_at_temperature(self.T_ambient)
        
        # Reynolds数
        Re = (air_props['rho'] * self.wind_speed * self.equipment_diameter) / air_props['mu']
        
        # Nusselt数
        Nu = ConvectionModel.churchill_bernstein_cylinder(Re, air_props['Pr'])
        
        # 对流系数
        h = ConvectionModel.nusselt_to_h(Nu, air_props['k'], self.equipment_diameter)
        
        return h
    
    def solve_steady_state(self, initial_guess=None):
        """
        求解稳态温度分布
        
        返回值:
            temperatures: 各层边界温度 [K]
            heat_flux: 热流 [W/m²]
        """
        n_layers = len(self.materials)
        
        if initial_guess is None:
            # 线性初始猜测
            initial_guess = np.linspace(self.T_gas, self.T_ambient, n_layers + 1)
        
        # 需要求解的未知数:内层界面温度(n_layers-1个)
        def equations(T_interfaces):
            # T_interfaces: 内层界面的温度
            # 完整温度列表
            T_all = np.zeros(n_layers + 1)
            T_all[0] = self.T_gas  # 热面(烟气侧)
            T_all[-1] = self.T_ambient  # 冷面(环境侧)
            T_all[1:-1] = T_interfaces
            
            # 确保温度单调递减
            for i in range(1, n_layers):
                if T_all[i] > T_all[i-1]:
                    T_all[i] = T_all[i-1] - 1  # 强制递减
                if T_all[i] < T_all[i+1]:
                    T_all[i] = T_all[i+1] + 1  # 强制递减
            
            residuals = []
            
            # 每层的热流应该相等(稳态)
            heat_fluxes = []
            
            # 烟气侧对流
            q_conv_gas = self.h_gas * (self.T_gas - T_all[0])
            heat_fluxes.append(q_conv_gas)
            
            # 每层导热
            for i in range(n_layers):
                T_hot = T_all[i]
                T_cold = T_all[i + 1]
                
                if abs(T_hot - T_cold) < 0.1:
                    k_mean = self.materials[i].thermal_conductivity((T_hot + T_cold) / 2)
                else:
                    k_mean = self.materials[i].thermal_conductivity_mean(T_hot, T_cold)
                
                q_cond = k_mean * (T_hot - T_cold) / self.materials[i].thickness
                heat_fluxes.append(q_cond)
            
            # 环境侧对流和辐射
            T_outer = T_all[-1]
            # 线性化的总体对流系数(对流+辐射)
            h_rad = RadiationModel.radiation_heat_transfer_coefficient(T_outer, self.T_ambient, 0.8)
            h_total = self.h_ambient + h_rad
            q_conv_ambient = h_total * (T_outer - self.T_ambient)
            heat_fluxes.append(q_conv_ambient)
            
            # 热流平衡方程
            for i in range(len(heat_fluxes) - 1):
                residuals.append(heat_fluxes[i] - heat_fluxes[i+1])
            
            return residuals
        
        # 求解
        T_interfaces_guess = initial_guess[1:-1]
        solution = fsolve(equations, T_interfaces_guess, full_output=True)
        T_interfaces = solution[0]
        info = solution[1]
        
        # 检查收敛
        if np.max(np.abs(info['fvec'])) > 1e-2:
            print(f"警告: 求解可能未完全收敛,残差={np.max(np.abs(info['fvec'])):e}")
        
        # 构建完整的温度列表
        temperatures = np.zeros(n_layers + 1)
        temperatures[0] = self.T_gas
        temperatures[-1] = self.T_ambient
        temperatures[1:-1] = T_interfaces
        
        # 计算最终热流
        q_final = self.h_gas * (self.T_gas - temperatures[0])
        
        return temperatures, q_final
    
    def calculate_thermal_resistance(self, temperatures, heat_flux):
        """
        计算热阻
        
        参数:
            temperatures: 温度分布
            heat_flux: 热流
        
        返回值:
            R_dict: 热阻字典
        """
        R_dict = {}
        
        # 烟气侧对流热阻
        R_dict['gas_convection'] = (self.T_gas - temperatures[0]) / heat_flux if heat_flux > 0 else 0
        
        # 每层的导热热阻
        for i, mat in enumerate(self.materials):
            T_hot = temperatures[i]
            T_cold = temperatures[i + 1]
            k_mean = mat.thermal_conductivity_mean(T_hot, T_cold)
            R_dict[f'{mat.name}_conduction'] = (T_hot - T_cold) / heat_flux if heat_flux > 0 else 0
        
        # 环境侧
        T_outer = temperatures[-1]
        h_rad = RadiationModel.radiation_heat_transfer_coefficient(T_outer, self.T_ambient, 0.8)
        h_total = self.h_ambient + h_rad
        R_dict['ambient_convection'] = (T_outer - self.T_ambient) / heat_flux if heat_flux > 0 else 0
        R_dict['ambient_h_total'] = h_total
        
        return R_dict
    
    def calculate_layer_properties(self, temperatures, heat_flux):
        """
        计算每层的详细参数
        """
        results = []
        
        for i, mat in enumerate(self.materials):
            T_hot = temperatures[i]
            T_cold = temperatures[i + 1]
            T_mean = (T_hot + T_cold) / 2
            k_mean = mat.thermal_conductivity_mean(T_hot, T_cold)
            
            result = {
                'name': mat.name,
                'thickness': mat.thickness * 1000,  # mm
                'T_hot': T_hot - 273.15,  # °C
                'T_cold': T_cold - 273.15,  # °C
                'T_mean': T_mean - 273.15,  # °C
                'k_mean': k_mean,
                'dT': T_hot - T_cold
            }
            results.append(result)
        
        return results


def plot_temperature_profile(materials, temperatures, save_fig=True):
    """
    绘制温度-厚度曲线
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('热工设备多层绝热结构温度分析', fontsize=16, fontweight='bold')
    
    # 1. 温度-厚度曲线
    ax1 = axes[0, 0]
    cumulative_thickness = np.cumsum([0] + [m.thickness * 1000 for m in materials])
    temperatures_celsius = temperatures - 273.15
    
    # 绘制阶梯曲线
    for i in range(len(materials)):
        x_start = cumulative_thickness[i]
        x_end = cumulative_thickness[i + 1]
        T_start = temperatures_celsius[i]
        T_end = temperatures_celsius[i + 1]
        
        ax1.plot([x_start, x_start], [T_start, T_start], 'o-', linewidth=2, markersize=8)
        ax1.plot([x_start, x_end], [T_start, T_start], 's-', linewidth=2.5, 
                label=f'{materials[i].name}', markersize=6)
        ax1.plot([x_end, x_end], [T_start, T_end], '-', linewidth=2, color='gray', alpha=0.7)
    
    ax1.plot(cumulative_thickness[-1], temperatures_celsius[-1], 'o', linewidth=2, markersize=8)
    
    ax1.set_xlabel('厚度累积距离 (mm)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('温度 (°C)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) 温度分布沿厚度方向', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    for i in range(len(materials)):
        ax1.axvline(x=cumulative_thickness[i+1], color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    # 2. 各层温度对比
    ax2 = axes[0, 1]
    layer_names = [m.name for m in materials]
    hot_temps = [temperatures_celsius[i] for i in range(len(materials))]
    cold_temps = [temperatures_celsius[i+1] for i in range(len(materials))]
    
    x_pos = np.arange(len(materials))
    width = 0.35
    
    ax2.bar(x_pos - width/2, hot_temps, width, label='热面温度', color='#FF6B6B', alpha=0.8)
    ax2.bar(x_pos + width/2, cold_temps, width, label='冷面温度', color='#4ECDC4', alpha=0.8)
    
    ax2.set_ylabel('温度 (°C)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) 各层的热面和冷面温度', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(layer_names, rotation=15, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 温度降分布
    ax3 = axes[1, 0]
    temperature_drops = [temperatures_celsius[i] - temperatures_celsius[i+1] for i in range(len(materials))]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax3.barh(layer_names, temperature_drops, color=colors[:len(materials)], alpha=0.8)
    
    ax3.set_xlabel('温度降 (°C)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) 各层的温度降', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    for i, val in enumerate(temperature_drops):
        ax3.text(val + 1, i, f'{val:.1f}°C', va='center', fontsize=9)
    
    # 4. 相对厚度和导热系数
    ax4 = axes[1, 1]
    thicknesses = [m.thickness * 1000 for m in materials]
    mean_temps = [(temperatures_celsius[i] + temperatures_celsius[i+1])/2 for i in range(len(materials))]
    
    ax4_left = ax4
    ax4_left.bar(x_pos, thicknesses, width=0.4, label='厚度', color='#95E1D3', alpha=0.8)
    ax4_left.set_ylabel('厚度 (mm)', fontsize=11, fontweight='bold', color='#95E1D3')
    ax4_left.tick_params(axis='y', labelcolor='#95E1D3')
    ax4_left.set_xticks(x_pos)
    ax4_left.set_xticklabels(layer_names, rotation=15, ha='right')
    
    ax4_right = ax4_left.twinx()
    k_values = [materials[i].thermal_conductivity((temperatures[i] + temperatures[i+1])/2) 
               for i in range(len(materials))]
    ax4_right.plot(x_pos, k_values, 'ro-', linewidth=2.5, markersize=8, label='导热系数')
    ax4_right.set_ylabel('导热系数 W/(m·K)', fontsize=11, fontweight='bold', color='red')
    ax4_right.tick_params(axis='y', labelcolor='red')
    
    ax4_left.set_title('(d) 厚度和导热系数对比', fontsize=12, fontweight='bold')
    ax4_left.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('thermal_analysis.png', dpi=300, bbox_inches='tight')
        print("\n图形已保存为 'thermal_analysis.png'")
    
    plt.show()


def print_results(temperatures, heat_flux, materials, layer_props, R_dict):
    """
    打印详细的计算结果
    """
    print("\n" + "="*70)
    print("多层绝热结构温度分布计算结果")
    print("="*70)
    
    print(f"\n热流密度: {heat_flux:.2f} W/m²")
    print(f"热流密度: {heat_flux/1e6:.2f} MW/m²")
    
    print("\n" + "-"*70)
    print("温度分布:")
    print("-"*70)
    print(f"{('位置'):<30} {('温度(°C)'):<20} {('温度(K)'):<20}")
    print("-"*70)
    print(f"{('烟气温度(热面外)'):<30} {temperatures[0]-273.15:>18.2f} {temperatures[0]:>18.2f}")
    
    for i, mat in enumerate(materials):
        print(f"\n{mat.name}:")
        print(f"  热面温度: {temperatures[i]-273.15:>18.2f}°C ({temperatures[i]:>10.2f}K)")
        print(f"  冷面温度: {temperatures[i+1]-273.15:>18.2f}°C ({temperatures[i+1]:>10.2f}K)")
    
    print(f"\n{('环境温度(冷面外)'):<30} {temperatures[-1]-273.15:>18.2f} {temperatures[-1]:>18.2f}")
    
    print("\n" + "-"*70)
    print("各层详细参数:")
    print("-"*70)
    print(f"{('材料'):<20} {('厚度(mm)'):<12} {('热面(°C)'):<12} {('冷面(°C)'):<12} {('导热系数W/(m·K)'):<18}")
    print("-"*70)
    
    for prop in layer_props:
        print(f"{prop['name']:<20} {prop['thickness']:>10.2f} "
              f"{prop['T_hot']:>10.2f} {prop['T_cold']:>10.2f} {prop['k_mean']:>16.4f}")
    
    print("\n" + "-"*70)
    print("热阻分析:")
    print("-"*70)
    total_R = 0
    print(f"{('热阻项'):<30} {('数值(K/W)'):>15} {('百分比(%)'):>15}")
    print("-"*70)
    
    for key, value in R_dict.items():
        if key != 'ambient_h_total' and isinstance(value, (int, float)):
            total_R += value
    
    for key, value in R_dict.items():
        if key != 'ambient_h_total' and isinstance(value, (int, float)):
            percentage = (value / total_R * 100) if total_R > 0 else 0
            print(f"{key:<30} {value:>15.6f} {percentage:>14.2f}")
    
    print("-"*70)
    print(f"{('总热阻'):<30} {total_R:>15.6f} {100.0:>14.2f}")
    print("="*70)
    
    # 外壁温度最重要的结果
    print(f"\n*** 外壁温度: {temperatures[-1]-273.15:.2f}°C ({temperatures[-1]:.2f}K) ***\n")


if __name__ == "__main__":
    # 示例:定义一个热工设备的多层绝热结构
    print("\n热工设备多层绝热结构温度计算")
    print("="*70)
    
    # 定义材料(从内到外)
    materials = [
        ThermalMaterial(
            name="高铝耐火砖",
            thickness=0.2,  # 200 mm
            k0=1.5,  # W/(m·K)
            a=0.001,  # 一次项系数
            b=0.0001,  # 二次项系数
            density=2800,
            specific_heat=960,
            emissivity=0.8
        ),
        ThermalMaterial(
            name="陶土保温棉",
            thickness=0.15,  # 150 mm
            k0=0.12,  # W/(m·K)
            a=0.002,
            b=0.00005,
            density=200,
            specific_heat=1050,
            emissivity=0.9
        ),
        ThermalMaterial(
            name="硅酸铝纤维",
            thickness=0.1,  # 100 mm
            k0=0.08,  # W/(m·K)
            a=0.001,
            b=0.00002,
            density=150,
            specific_heat=1100,
            emissivity=0.95
        )
    ]
    
    # 边界条件
    T_gas = 1273.15  # 1000°C = 1273.15K (烟气温度)
    T_ambient = 303.15  # 30°C = 303.15K (环境温度)
    wind_speed = 2.0  # m/s (风速)
    equipment_diameter = 1.0  # m (设备外径)
    
    print(f"\n输入参数:")
    print(f"  烟气温度: {T_gas - 273.15:.2f}°C")
    print(f"  环境温度: {T_ambient - 273.15:.2f}°C")
    print(f"  风速: {wind_speed:.2f} m/s")
    print(f"  设备外径: {equipment_diameter:.2f} m")
    
    # 创建分析器并求解
    analyzer = ThermalMultilayerAnalyzer(
        materials_list=materials,
        T_gas=T_gas,
        T_ambient=T_ambient,
        wind_speed=wind_speed,
        equipment_diameter=equipment_diameter
    )
    
    print("\n正在求解稳态温度分布...")
    temperatures, heat_flux = analyzer.solve_steady_state()
    
    # 计算热阻
    R_dict = analyzer.calculate_thermal_resistance(temperatures, heat_flux)
    
    # 计算每层详细参数
    layer_props = analyzer.calculate_layer_properties(temperatures, heat_flux)
    
    # 打印结果
    print_results(temperatures, heat_flux, materials, layer_props, R_dict)
    
    # 绘制温度-厚度曲线
    plot_temperature_profile(materials, temperatures)
