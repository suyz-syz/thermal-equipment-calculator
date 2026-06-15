#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热工设备多层绝热结构温度计算程序 (改进版)

核心改进:
1. 使用约束优化替代fsolve，提高稳定性
2. 烟气流速作为可配置参数
3. 修复热阻计算浮点精度问题
4. 使用材料本身的发射率
5. 添加烟气侧辐射
6. 优化空气物性计算
7. 改进字体处理
8. 添加单元测试

作者: Thermal Engineering Team
日期: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize, fsolve
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# 改进的字体设置 - 添加fallback机制
def setup_matplotlib_fonts():
    """
    设置matplotlib字体，支持多平台
    """
    fonts = {
        'darwin': ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'],  # macOS
        'linux': ['DejaVu Sans', 'Noto Sans CJK SC'],              # Linux
        'win32': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']      # Windows
    }
    
    import sys
    platform = sys.platform
    font_list = fonts.get(platform, ['DejaVu Sans'])
    
    try:
        rcParams['font.sans-serif'] = font_list
    except:
        rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    rcParams['axes.unicode_minus'] = False

setup_matplotlib_fonts()


class ThermalMaterial:
    """
    耐火/保温材料类
    
    导热系数采用二项式温度依赖模型:
    k(T) = k0 * (1 + a*T + b*T^2)
    其中T为相对温度: (T_abs - 300) / 300
    """
    
    def __init__(self, name, thickness, k0, a=0.0, b=0.0, 
                 density=1.0, specific_heat=1.0, emissivity=0.8):
        """
        初始化材料参数
        
        参数:
            name: 材料名称
            thickness: 材料厚度 [m]
            k0: 参考导热系数 [W/(m·K)] (在300K时)
            a: 一次项系数
            b: 二次项系数
            density: 密度 [kg/m³]
            specific_heat: 比热容 [J/(kg·K)]
            emissivity: 发射率 [-]，范围0.0-1.0
        """
        # 参数验证
        if thickness <= 0:
            raise ValueError(f"厚度必须为正数，得到: {thickness}")
        if k0 <= 0:
            raise ValueError(f"导热系数必须为正数，得到: {k0}")
        if not (0.0 <= emissivity <= 1.0):
            raise ValueError(f"发射率必须在0-1之间，得到: {emissivity}")
        
        self.name = name
        self.thickness = thickness
        self.k0 = k0
        self.a = a
        self.b = b
        self.density = density
        self.specific_heat = specific_heat
        self.emissivity = emissivity
        
        # 缓存空气物性以减少重复计算
        self._air_props_cache = {}
    
    def thermal_conductivity(self, T):
        """
        计算在温度T处的导热系数
        
        参数:
            T: 绝对温度 [K]
        
        返回值:
            导热系数 [W/(m·K)]
        """
        T_norm = (T - 300.0) / 300.0
        k = self.k0 * (1.0 + self.a * T_norm + self.b * T_norm**2)
        return max(k, 0.01)  # 防止导热系数为负
    
    def thermal_conductivity_mean(self, T1, T2):
        """
        计算两个温度之间的平均导热系数
        使用数值积分确保精度
        
        参数:
            T1, T2: 温度 [K]
        
        返回值:
            平均导热系数 [W/(m·K)]
        """
        def integrand(T):
            return self.thermal_conductivity(T)
        
        T_min, T_max = min(T1, T2), max(T1, T2)
        
        # 如果温度差过小，直接返回中点值
        if abs(T_max - T_min) < 0.5:
            return self.thermal_conductivity((T_min + T_max) / 2)
        
        try:
            k_mean, _ = quad(integrand, T_min, T_max, limit=100, epsabs=1e-8)
            k_mean /= (T_max - T_min)
            return k_mean
        except:
            # 数值积分失败时，使用梯形法则
            T_range = np.linspace(T_min, T_max, 10)
            k_vals = np.array([self.thermal_conductivity(T) for T in T_range])
            return np.trapz(k_vals, T_range) / (T_max - T_min)


class ConvectionModel:
    """
    对流和辐射换热模型
    """
    
    STEFAN_BOLTZMANN = 5.67e-8  # Stefan-Boltzmann常数 [W/(m²·K⁴)]
    
    @staticmethod
    def churchill_bernstein_cylinder(Re, Pr):
        """
        Churchill-Bernstein关系式(圆柱形横流)
        有效范围: 0.4 <= Re <= 4×10^5
        
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
            # 高Reynolds数
            Nu = 0.174 * Re**0.618 * Pr**0.33
        return Nu
    
    @staticmethod
    def nusselt_to_h(Nu, k_air, D):
        """
        从Nusselt数计算对流换热系数
        
        参数:
            Nu: Nusselt数
            k_air: 流体导热系数 [W/(m·K)]
            D: 特征长度 [m]
        
        返回值:
            对流换热系数 [W/(m²·K)]
        """
        if D <= 0:
            raise ValueError(f"特征长度必须为正数，得到: {D}")
        return Nu * k_air / D
    
    @classmethod
    def radiation_heat_transfer_coefficient(cls, T_surface, T_ambient, emissivity):
        """
        计算辐射换热的等效线性化系数
        h_rad = ε·σ·(T_s² + T_a²)·(T_s + T_a)
        
        参数:
            T_surface: 表面温度 [K]
            T_ambient: 环境温度 [K]
            emissivity: 发射率 [-]
        
        返回值:
            等效辐射换热系数 [W/(m²·K)]
        """
        if emissivity <= 0:
            return 0.0
        
        T_s2 = T_surface**2
        T_a2 = T_ambient**2
        h_rad = emissivity * cls.STEFAN_BOLTZMANN * (T_s2 + T_a2) * (T_surface + T_ambient)
        return max(h_rad, 0.0)


class AirProperties:
    """
    空气物性参数计算
    支持缓存以提高性能
    """
    
    _cache = {}  # 类级别缓存
    _CACHE_SIZE = 100
    
    @classmethod
    def properties_at_temperature(cls, T):
        """
        计算空气在温度T处的物性参数
        基于Sutherland公式和NIST数据
        
        参数:
            T: 温度 [K]
        
        返回值:
            字典，包含 rho, cp, k, mu, Pr
        """
        # 检查缓存
        T_rounded = round(T, -1)  # 舍入到最近的10K
        if T_rounded in cls._cache:
            return cls._cache[T_rounded]
        
        # 参考状态(300K, 101325 Pa)
        rho_ref = 1.177
        cp_ref = 1005.0
        k_ref = 0.0263
        mu_ref = 1.846e-5
        T_ref = 300.0
        
        # 理想气体密度变化
        rho = rho_ref * (T_ref / T)
        
        # 比热容随温度变化
        cp = 1000.0 + 0.1 * (T - T_ref) + 0.0001 * (T - T_ref)**2
        
        # Sutherland导热系数公式
        S_k = 194.0
        k = k_ref * ((T / T_ref)**1.5) * (T_ref + S_k) / (T + S_k)
        
        # Sutherland动力粘度公式
        S_mu = 110.0
        mu = mu_ref * ((T / T_ref)**1.5) * (T_ref + S_mu) / (T + S_mu)
        
        # Prandtl数
        Pr = (cp * mu) / k
        
        props = {
            'rho': rho,
            'cp': cp,
            'k': k,
            'mu': mu,
            'Pr': Pr
        }
        
        # 缓存结果
        if len(cls._cache) < cls._CACHE_SIZE:
            cls._cache[T_rounded] = props
        
        return props
    
    @classmethod
    def clear_cache(cls):
        """清除缓存"""
        cls._cache.clear()


class ThermalMultilayerAnalyzer:
    """
    多层绝热结构热传导分析器
    改进版本：采用约束优化提高稳定性
    """
    
    def __init__(self, materials_list, T_gas, T_ambient, h_gas=None, h_ambient=None,
                 v_gas=10.0, wind_speed=2.0, equipment_diameter=1.0,
                 include_radiation_gas_side=True):
        """
        初始化分析器
        
        参数:
            materials_list: 材料列表
            T_gas: 烟气温度 [K]
            T_ambient: 环境温度 [K]
            h_gas: 烟气侧对流系数 [W/(m²·K)]，如果为None则自动计算
            h_ambient: 环境侧对流系数 [W/(m²·K)]，如果为None则自动计算
            v_gas: 烟气流速 [m/s]，默认10 m/s
            wind_speed: 风速 [m/s]
            equipment_diameter: 设备外径 [m]
            include_radiation_gas_side: 是否考虑烟气侧辐射
        """
        # 参数验证
        if T_gas <= T_ambient:
            raise ValueError(f"烟气温度({T_gas}K)必须高于环境温度({T_ambient}K)")
        if v_gas <= 0:
            raise ValueError(f"烟气流速必须为正数，得到: {v_gas}")
        if not materials_list:
            raise ValueError("材料列表不能为空")
        
        self.materials = materials_list
        self.T_gas = T_gas
        self.T_ambient = T_ambient
        self.v_gas = v_gas
        self.wind_speed = max(wind_speed, 0.1)  # 最小风速0.1 m/s
        self.equipment_diameter = equipment_diameter
        self.include_radiation_gas_side = include_radiation_gas_side
        
        # 计算对流系数
        if h_gas is None:
            self.h_gas = self._calculate_gas_side_h()
        else:
            self.h_gas = h_gas
        
        if h_ambient is None:
            self.h_ambient = self._calculate_ambient_side_h()
        else:
            self.h_ambient = h_ambient
        
        print(f"\n=== 对流换热系数 ===")
        print(f"烟气流速: {self.v_gas:.2f} m/s")
        print(f"烟气侧对流系数: {self.h_gas:.2f} W/(m²·K)")
        print(f"环境侧对流系数: {self.h_ambient:.2f} W/(m²·K)")
        if self.include_radiation_gas_side:
            print(f"烟气侧已考虑辐射换热")
    
    def _calculate_gas_side_h(self):
        """
        计算烟气侧对流系数
        """
        T_mean = (self.T_gas + self.T_ambient) / 2
        air_props = AirProperties.properties_at_temperature(T_mean)
        
        Re = (air_props['rho'] * self.v_gas * self.equipment_diameter) / air_props['mu']
        Nu = ConvectionModel.churchill_bernstein_cylinder(Re, air_props['Pr'])
        h = ConvectionModel.nusselt_to_h(Nu, air_props['k'], self.equipment_diameter)
        
        return h
    
    def _calculate_ambient_side_h(self):
        """
        计算环境侧对流系数
        """
        air_props = AirProperties.properties_at_temperature(self.T_ambient)
        Re = (air_props['rho'] * self.wind_speed * self.equipment_diameter) / air_props['mu']
        Nu = ConvectionModel.churchill_bernstein_cylinder(Re, air_props['Pr'])
        h = ConvectionModel.nusselt_to_h(Nu, air_props['k'], self.equipment_diameter)
        
        return h
    
    def _calculate_heat_flux(self, T_interfaces):
        """
        计算给定温度分布下的热流和残差
        
        参数:
            T_interfaces: 内层界面温度数组
        
        返回值:
            heat_flux: 热流密度
            residuals: 残差平方和
        """
        n_layers = len(self.materials)
        
        # 构建完整温度数组
        T_all = np.zeros(n_layers + 1)
        T_all[0] = self.T_gas
        T_all[-1] = self.T_ambient
        T_all[1:-1] = T_interfaces
        
        # 确保单调递减
        for i in range(1, n_layers):
            T_all[i] = np.minimum(T_all[i], T_all[i-1] - 0.1)
            T_all[i] = np.maximum(T_all[i], T_all[i+1] + 0.1)
        
        heat_fluxes = []
        
        # 烟气侧对流 + 可选的辐射
        h_gas_total = self.h_gas
        if self.include_radiation_gas_side:
            h_rad_gas = ConvectionModel.radiation_heat_transfer_coefficient(
                T_all[0], self.T_gas, self.materials[0].emissivity)
            h_gas_total += h_rad_gas
        
        q_gas = h_gas_total * (self.T_gas - T_all[0])
        heat_fluxes.append(q_gas)
        
        # 各层导热
        for i in range(n_layers):
            T_hot = T_all[i]
            T_cold = T_all[i + 1]
            k_mean = self.materials[i].thermal_conductivity_mean(T_hot, T_cold)
            q_cond = k_mean * (T_hot - T_cold) / self.materials[i].thickness
            heat_fluxes.append(q_cond)
        
        # 环境侧对流 + 辐射
        T_outer = T_all[-1]
        h_rad_amb = ConvectionModel.radiation_heat_transfer_coefficient(
            T_outer, self.T_ambient, self.materials[-1].emissivity)
        h_total_amb = self.h_ambient + h_rad_amb
        q_amb = h_total_amb * (T_outer - self.T_ambient)
        heat_fluxes.append(q_amb)
        
        # 计算残差（热流平衡）
        heat_fluxes = np.array(heat_fluxes)
        residuals = np.sum((heat_fluxes[:-1] - heat_fluxes[1:]) ** 2)
        
        return heat_fluxes[-1], residuals, heat_fluxes
    
    def solve_steady_state(self, method='minimize', initial_guess=None):
        """
        求解稳态温度分布
        
        参数:
            method: 求解方法 ('minimize' 或 'fsolve')
            initial_guess: 初始猜测
        
        返回值:
            temperatures: 各层边界温度
            heat_flux: 热流
            success: 是否收敛成功
        """
        n_layers = len(self.materials)
        
        if initial_guess is None:
            # 线性初始猜测
            initial_guess = np.linspace(self.T_gas, self.T_ambient, n_layers + 1)
        
        T_interfaces_guess = initial_guess[1:-1]
        
        if method == 'minimize':
            # 使用约束优化（更稳定）
            result = minimize(
                lambda T_int: self._calculate_heat_flux(T_int)[1],
                T_interfaces_guess,
                method='L-BFGS-B',
                bounds=[(self.T_ambient + 0.1, self.T_gas - 0.1) for _ in T_interfaces_guess],
                options={'ftol': 1e-10, 'maxiter': 500}
            )
            
            success = result.success
            T_interfaces = result.x
            
        else:  # fsolve方法
            def equations(T_int):
                _, _, fluxes = self._calculate_heat_flux(T_int)
                return fluxes[:-1] - fluxes[1:]
            
            T_interfaces, info, ier, msg = fsolve(
                equations, T_interfaces_guess, full_output=True
            )
            success = ier == 1
        
        # 构建完整温度数组
        temperatures = np.zeros(n_layers + 1)
        temperatures[0] = self.T_gas
        temperatures[-1] = self.T_ambient
        temperatures[1:-1] = T_interfaces
        
        heat_flux, residual, _ = self._calculate_heat_flux(T_interfaces)
        
        if not success or residual > 1.0:
            print(f"警告: 求解收敛性较差，残差={residual:.2e}")
        
        return temperatures, heat_flux, success
    
    def calculate_thermal_resistance(self, temperatures, heat_flux):
        """
        计算热阻（改进的浮点精度处理）
        """
        R_dict = {}
        n_layers = len(self.materials)
        
        if heat_flux <= 0:
            return {f'ERROR': 'Invalid heat flux'}
        
        # 烟气侧
        h_gas_total = self.h_gas
        if self.include_radiation_gas_side:
            h_rad_gas = ConvectionModel.radiation_heat_transfer_coefficient(
                temperatures[0], self.T_gas, self.materials[0].emissivity)
            h_gas_total += h_rad_gas
        
        R_dict['gas_side'] = (self.T_gas - temperatures[0]) / heat_flux
        
        # 各层导热
        for i, mat in enumerate(self.materials):
            T_hot = temperatures[i]
            T_cold = temperatures[i + 1]
            k_mean = mat.thermal_conductivity_mean(T_hot, T_cold)
            R_dict[f'Layer_{i+1}_{mat.name}'] = (T_hot - T_cold) / heat_flux
        
        # 环境侧
        T_outer = temperatures[-1]
        h_rad_amb = ConvectionModel.radiation_heat_transfer_coefficient(
            T_outer, self.T_ambient, self.materials[-1].emissivity)
        R_dict['ambient_side'] = (T_outer - self.T_ambient) / heat_flux
        
        return R_dict
    
    def calculate_layer_properties(self, temperatures, heat_flux):
        """
        计算每层的详细参数
        """
        results = []
        for i, mat in enumerate(self.materials):
            T_hot = temperatures[i]
            T_cold = temperatures[i + 1]
            k_mean = mat.thermal_conductivity_mean(T_hot, T_cold)
            
            result = {
                'name': mat.name,
                'thickness': mat.thickness * 1000,
                'T_hot': T_hot - 273.15,
                'T_cold': T_cold - 273.15,
                'T_mean': (T_hot + T_cold) / 2 - 273.15,
                'k_mean': k_mean,
                'dT': T_hot - T_cold,
                'emissivity': mat.emissivity
            }
            results.append(result)
        return results


def plot_temperature_profile(materials, temperatures, save_fig=True):
    """
    绘制温度-厚度曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('热工设备多层绝热结构温度分析', fontsize=16, fontweight='bold')
    
    cumulative_thickness = np.cumsum([0] + [m.thickness * 1000 for m in materials])
    temperatures_celsius = temperatures - 273.15
    
    # 1. 温度-厚度曲线
    ax1 = axes[0, 0]
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
    
    # 2. 各层温度对比
    ax2 = axes[0, 1]
    layer_names = [m.name for m in materials]
    hot_temps = [temperatures_celsius[i] for i in range(len(materials))]
    cold_temps = [temperatures_celsius[i+1] for i in range(len(materials))]
    
    x_pos = np.arange(len(materials))
    width = 0.35
    ax2.bar(x_pos - width/2, hot_temps, width, label='热面', color='#FF6B6B', alpha=0.8)
    ax2.bar(x_pos + width/2, cold_temps, width, label='冷面', color='#4ECDC4', alpha=0.8)
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
    
    # 4. 厚度和导热系数
    ax4 = axes[1, 1]
    thicknesses = [m.thickness * 1000 for m in materials]
    ax4_left = ax4
    ax4_left.bar(x_pos, thicknesses, width=0.4, label='厚度', color='#95E1D3', alpha=0.8)
    ax4_left.set_ylabel('厚度 (mm)', fontsize=11, fontweight='bold', color='#95E1D3')
    ax4_left.set_xticks(x_pos)
    ax4_left.set_xticklabels(layer_names, rotation=15, ha='right')
    
    ax4_right = ax4_left.twinx()
    k_values = [materials[i].thermal_conductivity((temperatures[i] + temperatures[i+1])/2)
               for i in range(len(materials))]
    ax4_right.plot(x_pos, k_values, 'ro-', linewidth=2.5, markersize=8, label='导热系数')
    ax4_right.set_ylabel('导热系数 W/(m·K)', fontsize=11, fontweight='bold', color='red')
    
    ax4_left.set_title('(d) 厚度和导热系数对比', fontsize=12, fontweight='bold')
    ax4_left.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_fig:
        plt.savefig('thermal_analysis.png', dpi=300, bbox_inches='tight')
        print("图形已保存为 'thermal_analysis.png'")
    plt.show()


def print_results(temperatures, heat_flux, materials, layer_props, R_dict):
    """
    打印计算结果（改进的热阻计算）
    """
    print("\n" + "="*70)
    print("多层绝热结构温度分布计算结果")
    print("="*70)
    
    print(f"\n热流密度: {heat_flux:.2f} W/m²")
    print(f"热流密度: {heat_flux/1e6:.4f} MW/m²")
    
    print("\n" + "-"*70)
    print("温度分布:")
    print("-"*70)
    print(f"{('位置'):<30} {('温度(°C)'):<20} {('温度(K)'):<20}")
    print("-"*70)
    
    for i, mat in enumerate(materials):
        print(f"\n{mat.name} (发射率: {mat.emissivity:.2f}):")
        print(f"  热面: {temperatures[i]-273.15:>18.2f}°C ({temperatures[i]:>10.2f}K)")
        print(f"  冷面: {temperatures[i+1]-273.15:>18.2f}°C ({temperatures[i+1]:>10.2f}K)")
    
    print("\n" + "-"*70)
    print("各层详细参数:")
    print("-"*70)
    print(f"{('材料'):<20} {('厚度mm'):<12} {('热面°C'):<12} {('冷面°C'):<12} {('导热W/mK'):<15}")
    print("-"*70)
    
    for prop in layer_props:
        print(f"{prop['name']:<20} {prop['thickness']:>10.2f} "
              f"{prop['T_hot']:>10.2f} {prop['T_cold']:>10.2f} {prop['k_mean']:>13.4f}")
    
    print("\n" + "-"*70)
    print("热阻分析:")
    print("-"*70)
    
    # 修复: 预先计算总热阻
    total_R = sum(v for k, v in R_dict.items() if isinstance(v, (int, float)))
    
    print(f"{('热阻项'):<30} {('数值(K/W)'):>15} {('百分比(%)'):>15}")
    print("-"*70)
    
    for key, value in R_dict.items():
        if isinstance(value, (int, float)):
            percentage = (value / total_R * 100) if total_R > 0 else 0
            print(f"{key:<30} {value:>15.6f} {percentage:>15.2f}")
    
    print("-"*70)
    print(f"{('总热阻'):<30} {total_R:>15.6f} {100.0:>15.2f}")
    print("="*70)
    
    print(f"\n*** 外壳温度: {temperatures[-1]-273.15:.2f}°C ({temperatures[-1]:.2f}K) ***\n")


# ============================================================================
# 单元测试
# ============================================================================

def run_unit_tests():
    """
    运行单元测试
    """
    print("\n" + "="*70)
    print("运行单元测试")
    print("="*70)
    
    # 测试1: 材料参数验证
    print("\n[测试1] 材料参数验证...")
    try:
        # 无效厚度
        mat = ThermalMaterial("test", -0.1, 1.0)
        print("  FAIL: 应该拒绝负厚度")
    except ValueError:
        print("  PASS: 正确拒绝负厚度")
    
    try:
        # 无效发射率
        mat = ThermalMaterial("test", 0.1, 1.0, emissivity=1.5)
        print("  FAIL: 应该拒绝发射率>1")
    except ValueError:
        print("  PASS: 正确拒绝发射率>1")
    
    # 测试2: 导热系数计算
    print("\n[测试2] 导热系数温度依赖性...")
    mat = ThermalMaterial("test", 0.1, 1.0, a=0.001, b=0.0001)
    k_300 = mat.thermal_conductivity(300)
    k_600 = mat.thermal_conductivity(600)
    if k_600 > k_300:
        print(f"  PASS: k(600K)={k_600:.4f} > k(300K)={k_300:.4f}")
    else:
        print(f"  FAIL: 导热系数应该随温度增加")
    
    # 测试3: 对流系数计算
    print("\n[测试3] Churchill-Bernstein相关式...")
    Nu = ConvectionModel.churchill_bernstein_cylinder(1000, 0.7)
    print(f"  Nu(Re=1000, Pr=0.7) = {Nu:.2f}")
    if Nu > 0:
        print("  PASS")
    else:
        print("  FAIL")
    
    # 测试4: 完整计算
    print("\n[测试4] 完整热力分析...")
    materials = [
        ThermalMaterial("Layer1", 0.1, 1.0, a=0.001, emissivity=0.8),
        ThermalMaterial("Layer2", 0.1, 0.1, a=0.002, emissivity=0.9),
    ]
    
    analyzer = ThermalMultilayerAnalyzer(
        materials_list=materials,
        T_gas=1273.15,
        T_ambient=303.15,
        v_gas=10.0,
        wind_speed=2.0,
        include_radiation_gas_side=True
    )
    
    temperatures, heat_flux, success = analyzer.solve_steady_state(method='minimize')
    
    if success and heat_flux > 0 and temperatures[-1] < temperatures[0]:
        print(f"  PASS: 热流={heat_flux:.2f} W/m², 外壳温度={temperatures[-1]-273.15:.2f}°C")
    else:
        print(f"  FAIL: 求解失败或结果不合理")
    
    print("\n" + "="*70)
    print("单元测试完成")
    print("="*70)


if __name__ == "__main__":
    # 运行单元测试
    run_unit_tests()
    
    print("\n\n" + "="*70)
    print("热工设备多层绝热结构温度计算 - 改进版")
    print("="*70)
    
    # 定义材料
    materials = [
        ThermalMaterial(
            name="高铝耐火砖",
            thickness=0.2,
            k0=1.5,
            a=0.001,
            b=0.0001,
            emissivity=0.8
        ),
        ThermalMaterial(
            name="陶土保温棉",
            thickness=0.15,
            k0=0.12,
            a=0.002,
            b=0.00005,
            emissivity=0.9
        ),
        ThermalMaterial(
            name="硅酸铝纤维",
            thickness=0.1,
            k0=0.08,
            a=0.001,
            b=0.00002,
            emissivity=0.95
        )
    ]
    
    # 边界条件
    T_gas = 1273.15  # 1000°C
    T_ambient = 303.15  # 30°C
    v_gas = 10.0  # 烟气流速 m/s
    wind_speed = 2.0  # 风速 m/s
    
    print(f"\n输入参数:")
    print(f"  烟气温度: {T_gas-273.15:.2f}°C")
    print(f"  环境温度: {T_ambient-273.15:.2f}°C")
    print(f"  烟气流速: {v_gas:.2f} m/s")
    print(f"  风速: {wind_speed:.2f} m/s")
    
    # 创建分析器
    analyzer = ThermalMultilayerAnalyzer(
        materials_list=materials,
        T_gas=T_gas,
        T_ambient=T_ambient,
        v_gas=v_gas,
        wind_speed=wind_speed,
        include_radiation_gas_side=True
    )
    
    print("\n求解稳态温度分布...")
    temperatures, heat_flux, success = analyzer.solve_steady_state(method='minimize')
    
    if success:
        print("✓ 求解成功")
    else:
        print("⚠ 求解收敛性可能不理想，但已得到解")
    
    R_dict = analyzer.calculate_thermal_resistance(temperatures, heat_flux)
    layer_props = analyzer.calculate_layer_properties(temperatures, heat_flux)
    
    print_results(temperatures, heat_flux, materials, layer_props, R_dict)
    
    # 绘图
    plot_temperature_profile(materials, temperatures)
