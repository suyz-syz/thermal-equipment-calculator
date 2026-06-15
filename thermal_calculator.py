"""
Enhanced Thermal Multilayer Wall Calculator
============================================
Solves steady-state heat conduction through multilayer walls with:
- Convection boundary conditions (烟气侧对流，环境侧对流+辐射)
- Temperature-dependent material properties
- Support for planar and cylindrical geometries
- Iterative solver (避免 fsolve 的 hack)
- Comprehensive physical validation

Reference:
  - ASME Handbook: Heat Transfer
  - Chen et al: Temperature-Dependent Thermal Properties
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

# ============================================================================
# Constants and Configuration
# ============================================================================

@dataclass
class ThermalConfig:
    """Global thermal calculation configuration."""
    CONVERGENCE_TOL: float = 1e-4  # Temperature convergence (K)
    MAX_ITERATIONS: int = 100  # Maximum solver iterations
    MIN_CONDUCTIVITY: float = 0.01  # Minimum k to avoid division issues
    REF_TEMPERATURE: float = 300.0  # Reference temperature for k normalization (K)
    GRAVITY: float = 9.81  # m/s^2
    STEFAN_BOLTZMANN: float = 5.67e-8  # W/m^2/K^4
    
    # Geometry support
    GEOMETRY_PLANAR: str = "planar"
    GEOMETRY_CYLINDRICAL: str = "cylindrical"


# ============================================================================
# Material Properties
# ============================================================================

class ThermalMaterial:
    """
    Represents a material with temperature-dependent thermal conductivity.
    
    k(T) is modeled as: k(T) = k_0 * (1 + a*(T - T_ref) + b*(T - T_ref)^2)
    
    Typical parameter ranges (工程材料):
      - 高温保温砖: a=1e-4, b=1e-7 (k0 ≈ 0.2-0.5 W/m/K)
      - 普通混凝土: a=5e-4, b=0 (k0 ≈ 1.4 W/m/K)
      - 陶土砖: a=2e-4, b=5e-8 (k0 ≈ 0.8 W/m/K)
    """
    
    def __init__(
        self,
        name: str,
        thickness: float,
        k0: float,
        rho: float,
        cp: float,
        a: float = 0.0,
        b: float = 0.0,
        T_ref: float = ThermalConfig.REF_TEMPERATURE,
    ):
        """
        Initialize thermal material.
        
        Args:
            name: Material identifier
            thickness: Layer thickness [m]
            k0: Reference thermal conductivity at T_ref [W/m/K]
            rho: Density [kg/m^3]
            cp: Specific heat [J/kg/K]
            a: Linear temperature coefficient [1/K]
            b: Quadratic temperature coefficient [1/K^2]
            T_ref: Reference temperature [K] (default 300 K)
        """
        self.name = name
        self.thickness = float(thickness)
        self.k0 = float(k0)
        self.rho = float(rho)
        self.cp = float(cp)
        self.a = float(a)
        self.b = float(b)
        self.T_ref = float(T_ref)
        
        # Validation
        if self.thickness <= 0:
            raise ValueError(f"Thickness must be positive, got {self.thickness}")
        if self.k0 <= 0:
            raise ValueError(f"k0 must be positive, got {self.k0}")
    
    def conductivity_at(self, T: float) -> float:
        """
        Compute thermal conductivity at temperature T.
        
        Args:
            T: Temperature [K]
            
        Returns:
            Thermal conductivity [W/m/K]
        """
        dT = T - self.T_ref
        k = self.k0 * (1.0 + self.a * dT + self.b * dT**2)
        return max(k, ThermalConfig.MIN_CONDUCTIVITY)
    
    def average_conductivity(self, T1: float, T2: float) -> float:
        """
        Compute average thermal conductivity over temperature range [T1, T2].
        Uses numerical integration (more accurate than linear average).
        
        Args:
            T1, T2: Temperature range [K]
            
        Returns:
            Average thermal conductivity [W/m/K]
        """
        if abs(T1 - T2) < 1e-6:
            return self.conductivity_at(T1)
        
        T_min, T_max = min(T1, T2), max(T1, T2)
        
        def integrand(T):
            return self.conductivity_at(T)
        
        k_avg, _ = quad(integrand, T_min, T_max, limit=50)
        k_avg /= (T_max - T_min)
        
        return k_avg
    
    def __repr__(self) -> str:
        return f"ThermalMaterial(name={self.name}, t={self.thickness:.3f}m, k0={self.k0:.3f})"


# ============================================================================
# Air Properties (烟气/环境空气)
# ============================================================================

class AirProperties:
    """
    Temperature-dependent properties of air/flue gas at atmospheric pressure.
    Uses Sutherland formula for viscosity and polynomial fits for other properties.
    
    Valid range: 300-2000 K (covers furnace/boiler applications)
    """
    
    # Sutherland formula coefficients for viscosity (空气)
    MU0 = 1.716e-5  # Pa·s at T0
    T0_MU = 273.15  # K
    S_MU = 110.4    # K (Sutherland constant)
    
    # Density at reference condition (ideal gas, 1 atm)
    RHO_REF = 1.225  # kg/m^3 at 288.15 K, 1 atm
    T_REF = 288.15   # K
    
    def __init__(self, pressure_pa: float = 101325.0):
        """
        Initialize air properties at given pressure.
        
        Args:
            pressure_pa: Pressure [Pa] (default 101325 = 1 atm)
        """
        self.pressure = float(pressure_pa)
    
    @staticmethod
    def viscosity(T: float) -> float:
        """
        Dynamic viscosity using Sutherland formula [Pa·s].
        
        μ(T) = μ0 * sqrt(T0_K/T) * (T + S) / (T0_K + S)
        
        Args:
            T: Temperature [K]
            
        Returns:
            Dynamic viscosity [Pa·s]
        """
        sqrt_ratio = np.sqrt(AirProperties.T0_MU / T)
        return (AirProperties.MU0 * sqrt_ratio * 
                (T + AirProperties.S_MU) / 
                (AirProperties.T0_MU + AirProperties.S_MU))
    
    @staticmethod
    def kinematic_viscosity(T: float, rho: float) -> float:
        """Kinematic viscosity [m^2/s]."""
        return AirProperties.viscosity(T) / rho
    
    @staticmethod
    def density(T: float, P: float = 101325.0) -> float:
        """
        Density using ideal gas law [kg/m^3].
        
        ρ = ρ_ref * (T_ref / T) * (P / P_ref)
        
        Args:
            T: Temperature [K]
            P: Pressure [Pa]
            
        Returns:
            Density [kg/m^3]
        """
        return AirProperties.RHO_REF * (AirProperties.T_REF / T) * (P / 101325.0)
    
    @staticmethod
    def specific_heat(T: float) -> float:
        """
        Specific heat at constant pressure [J/kg/K].
        Polynomial fit (600-1500 K typical furnace range).
        
        cp(T) = c0 + c1*T + c2*T^2 + c3*T^3
        
        Args:
            T: Temperature [K]
            
        Returns:
            Specific heat [J/kg/K]
        """
        # Polynomial coefficients fitted to NIST data (air, 600-1500 K)
        c = [1006.0, 0.148e-3, -0.235e-6, 0.158e-10]
        T_norm = T / 1000.0  # Normalize for numerical stability
        cp = c[0] + c[1]*T_norm + c[2]*T_norm**2 + c[3]*T_norm**3
        return max(cp, 700.0)  # Clamp to physical minimum
    
    @staticmethod
    def thermal_conductivity(T: float) -> float:
        """
        Thermal conductivity [W/m/K].
        Fits NIST data for air at atmospheric pressure (600-1500 K).
        
        Args:
            T: Temperature [K]
            
        Returns:
            Thermal conductivity [W/m/K]
        """
        # Empirical fit (high-temperature air)
        k0 = 0.025  # W/m/K at 300 K
        a = 0.0007  # Temperature coefficient [1/K]
        return k0 * (1.0 + a * (T - 300.0))
    
    @staticmethod
    def prandtl_number(T: float) -> float:
        """
        Prandtl number (dimensionless).
        
        Pr = cp * μ / k
        
        Args:
            T: Temperature [K]
            
        Returns:
            Prandtl number
        """
        mu = AirProperties.viscosity(T)
        cp = AirProperties.specific_heat(T)
        k = AirProperties.thermal_conductivity(T)
        return (cp * mu) / k


# ============================================================================
# Convection and Radiation Models
# ============================================================================

class ConvectionModel:
    """
    Heat transfer by convection using Churchill-Bernstein correlation.
    Applicable to cylinders in cross-flow (also used for planar approximation).
    """
    
    @staticmethod
    def churchill_bernstein(
        Re: float,
        Pr: float,
        geometry: str = ThermalConfig.GEOMETRY_PLANAR
    ) -> float:
        """
        Churchill-Bernstein Nusselt correlation for circular cylinders in cross-flow.
        
        For Re > 0.1 and all Pr:
        Nu = [0.3 + (0.62*Re^0.5*Pr^(1/3)) / (1 + (282000/Re)^(5/8))^(4/5)]^2
        
        For planar geometry, use equivalent diameter approximation.
        
        Args:
            Re: Reynolds number
            Pr: Prandtl number
            geometry: "planar" or "cylindrical"
            
        Returns:
            Nusselt number
        """
        # Avoid floating-point issues
        Re = max(Re, 0.1)
        Pr = max(Pr, 0.5)
        
        # Churchill-Bernstein formula
        Re_sqrt = np.sqrt(Re)
        Pr_cbrt = Pr ** (1.0 / 3.0)
        
        numerator = 0.62 * Re_sqrt * Pr_cbrt
        denominator = (1.0 + (282000.0 / Re) ** (5.0 / 8.0)) ** (4.0 / 5.0)
        
        Nu = (0.3 + numerator / denominator) ** 2
        
        return Nu
    
    @staticmethod
    def heat_transfer_coefficient(
        Nu: float,
        k_fluid: float,
        L_char: float
    ) -> float:
        """
        Compute convective heat transfer coefficient from Nusselt number.
        
        h = Nu * k / L_c
        
        Args:
            Nu: Nusselt number
            k_fluid: Fluid thermal conductivity [W/m/K]
            L_char: Characteristic length [m]
            
        Returns:
            Heat transfer coefficient [W/m^2/K]
        """
        if L_char <= 0:
            raise ValueError(f"Characteristic length must be positive")
        h = Nu * k_fluid / L_char
        return max(h, 0.1)  # Minimum h to avoid numerical issues


class RadiationModel:
    """
    Radiation heat transfer using linearized coefficients.
    
    q_rad = h_rad * (T_hot - T_cold)
    h_rad = ε * σ * (T_hot^2 + T_cold^2) * (T_hot + T_cold)
    """
    
    @staticmethod
    def radiation_coefficient(
        T_hot: float,
        T_cold: float,
        emissivity: float = 0.9,
        view_factor: float = 1.0
    ) -> float:
        """
        Linearized radiation heat transfer coefficient.
        
        Args:
            T_hot: Hot surface temperature [K]
            T_cold: Cold surface temperature [K]
            emissivity: Surface emissivity [0-1]
            view_factor: View factor [0-1]
            
        Returns:
            Linearized radiation coefficient [W/m^2/K]
        """
        sigma = ThermalConfig.STEFAN_BOLTZMANN
        eps = float(emissivity)
        F = float(view_factor)
        
        # Linearization: h_rad based on average of the two temperatures
        T_avg = 0.5 * (T_hot + T_cold)
        
        h_rad = eps * F * sigma * (T_hot + T_cold) * (T_hot**2 + T_cold**2) / (T_hot + T_cold)
        
        # Simplified: use average temperature
        h_rad = eps * F * sigma * 4.0 * T_avg**3
        
        return max(h_rad, 0.01)


# ============================================================================
# Thermal Analyzer (Core Solver)
# ============================================================================

class ThermalMultilayerAnalyzer:
    """
    Steady-state multilayer thermal analysis using iterative method.
    
    Physics:
    --------
    For N layers with convective boundaries:
    
    烟气侧:  q = h_gas * (T_gas - T_inner_1)
    Layer i: q = k_avg_i / d_i * (T_i - T_{i+1})
    环境侧:  q = h_env * (T_inner_N - T_env) + h_rad * (T_inner_N^4 - T_sky^4)
    
    Solution method: Fixed-point iteration
    1. Guess heat flux q
    2. Forward sweep: compute all temperatures from q
    3. Check boundary condition at environment side
    4. Update q, repeat until convergence
    """
    
    def __init__(
        self,
        materials: List[ThermalMaterial],
        geometry: str = ThermalConfig.GEOMETRY_PLANAR,
        inner_diameter: float = 1.0,
        outer_diameter: float = 2.0,
        length: float = 1.0,
        **kwargs
    ):
        """
        Initialize thermal analyzer.
        
        Args:
            materials: List of ThermalMaterial objects (inner to outer)
            geometry: "planar" or "cylindrical"
            inner_diameter: Inner diameter [m] (for cylindrical)
            outer_diameter: Outer diameter [m] (for cylindrical)
            length: Length [m] (for cylindrical)
            **kwargs: Additional solver parameters
        """
        self.materials = materials
        self.geometry = geometry
        self.inner_diameter = float(inner_diameter)
        self.outer_diameter = float(outer_diameter)
        self.length = float(length)
        
        # Solver configuration
        self.config = ThermalConfig()
        self.config.CONVERGENCE_TOL = kwargs.get("convergence_tol", 1e-4)
        self.config.MAX_ITERATIONS = kwargs.get("max_iterations", 100)
        
        # Geometry validation
        if self.geometry == ThermalConfig.GEOMETRY_CYLINDRICAL:
            if self.inner_diameter <= 0 or self.outer_diameter <= self.inner_diameter:
                raise ValueError("Invalid cylindrical geometry dimensions")
        
        # Last solution
        self.last_solution = None
    
    def compute_thermal_resistance(
        self,
        material: ThermalMaterial,
        T_inner: float,
        T_outer: float,
        area_factor: float = 1.0
    ) -> float:
        """
        Compute conduction thermal resistance for a layer.
        
        For planar: R = L / (k_avg * A)
        For cylindrical: R = ln(r_out/r_in) / (2*π*k_avg*L)
        
        Args:
            material: ThermalMaterial object
            T_inner, T_outer: Boundary temperatures [K]
            area_factor: Effective area factor
            
        Returns:
            Thermal resistance [K/W]
        """
        k_avg = material.average_conductivity(T_inner, T_outer)
        
        if self.geometry == ThermalConfig.GEOMETRY_PLANAR:
            # R = L / k
            R = material.thickness / k_avg / area_factor
        elif self.geometry == ThermalConfig.GEOMETRY_CYLINDRICAL:
            # Log-mean resistance for cylindrical coordinates
            r_in = self.inner_diameter / 2.0
            r_out = r_in + material.thickness
            ln_ratio = np.log(r_out / r_in)
            R = ln_ratio / (2.0 * np.pi * k_avg * self.length)
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}")
        
        return max(R, 1e-6)  # Avoid zero resistance
    
    def solve_steady_state_iterative(
        self,
        T_gas: float,
        h_gas: float,
        T_env: float,
        h_env: float,
        emissivity_outer: float = 0.9,
        T_sky: float = 300.0,
        verbose: bool = False
    ) -> Dict:
        """
        Solve steady-state using fixed-point iteration (更稳定的方法).
        
        Algorithm:
        ----------
        1. Estimate q from lumped resistance
        2. Forward sweep: compute T at each interface
        3. Compute T_outer from energy balance
        4. Check boundary condition
        5. Update q and repeat
        
        Args:
            T_gas: Flue gas temperature [K]
            h_gas: Convection coefficient (gas side) [W/m^2/K]
            T_env: Environment temperature [K]
            h_env: Convection coefficient (environment side) [W/m^2/K]
            emissivity_outer: Outer surface emissivity
            T_sky: Sky temperature for radiation [K]
            verbose: Print iteration details
            
        Returns:
            Dictionary with temperatures, fluxes, and resistances
        """
        n_layers = len(self.materials)
        config = self.config
        
        # Step 1: Initial heat flux estimate (lumped model)
        R_total_est = sum(
            material.thickness / material.k0 
            for material in self.materials
        )
        q_initial = (T_gas - T_env) / (1/h_gas + R_total_est + 1/h_env + 0.01)
        q = max(q_initial, 1.0)  # Start with positive flux
        
        if verbose:
            print(f"\n{'Iter':>4} | {'q [W/m²]':>12} | {'T_in [K]':>10} | {'T_out [K]':>10} | "
                  f"{'ΔT_outer [K]':>12} | {'Residual':>12}")
            print("-" * 75)
        
        # Iterative solve
        for iteration in range(config.MAX_ITERATIONS):
            # Step 2: Forward sweep - compute temperatures given q
            T = [0.0] * (n_layers + 1)  # T[0] = T_inner, T[n_layers] = T_outer
            
            # Inner surface temperature (烟气侧对流)
            T[0] = T_gas - q / h_gas
            
            # Temperature at each layer interface
            for i in range(n_layers):
                T_prev = T[i]
                T_next_est = T_prev - 1.0  # Initial guess for next layer
                
                # Refine using material resistance
                R_i = self.compute_thermal_resistance(
                    self.materials[i],
                    T_prev,
                    T_next_est
                )
                T[i + 1] = T_prev - q * R_i
            
            T_inner = T[0]
            T_outer = T[n_layers]
            
            # Step 3: Check outer boundary condition
            # Convection + Radiation
            h_rad = RadiationModel.radiation_coefficient(
                T_outer, T_sky, emissivity_outer, 1.0
            )
            q_outer = h_env * (T_outer - T_env) + h_rad * (T_outer - T_sky)
            
            # Residual
            residual = abs(q - q_outer)
            
            if verbose and iteration % 5 == 0:
                print(f"{iteration:4d} | {q:12.3f} | {T_inner:10.2f} | "
                      f"{T_outer:10.2f} | {q - q_outer:12.3f} | {residual:12.3e}")
            
            # Step 4: Check convergence
            if residual < config.CONVERGENCE_TOL:
                if verbose:
                    print(f"✓ Converged at iteration {iteration}")
                break
            
            # Step 5: Update heat flux (under-relaxation for stability)
            alpha = 0.5  # Under-relaxation factor
            q_new = alpha * q_outer + (1 - alpha) * q
            q = max(q_new, 0.1)  # Ensure positive
        
        else:
            if verbose:
                print(f"⚠ Warning: Did not converge after {config.MAX_ITERATIONS} iterations")
        
        # Store last solution
        result = {
            "heat_flux": q,
            "T_inner": T_inner,
            "T_outer": T_outer,
            "T_layers": T,
            "h_gas": h_gas,
            "h_env": h_env,
            "h_rad": h_rad,
            "iterations": iteration + 1,
            "residual": residual,
        }
        
        # Compute resistances
        R_conv_gas = 1.0 / h_gas
        R_total_cond = 0.0
        R_cond_layers = []
        
        for i in range(n_layers):
            R_i = self.compute_thermal_resistance(
                self.materials[i],
                T[i],
                T[i + 1]
            )
            R_cond_layers.append(R_i)
            R_total_cond += R_i
        
        R_conv_env = 1.0 / h_env
        R_rad_env = 1.0 / h_rad if h_rad > 0 else float('inf')
        
        result["R_conv_gas"] = R_conv_gas
        result["R_cond_layers"] = R_cond_layers
        result["R_total_cond"] = R_total_cond
        result["R_conv_env"] = R_conv_env
        result["R_rad_env"] = R_rad_env
        
        self.last_solution = result
        return result
    
    def compute_sensitivity(
        self,
        T_gas: float,
        h_gas: float,
        T_env: float,
        h_env: float,
        parameter_ranges: Dict[str, Tuple[float, float]],
        n_points: int = 5,
        verbose: bool = False
    ) -> Dict:
        """
        Sensitivity analysis: vary parameters and observe T_outer change.
        
        Args:
            T_gas, h_gas, T_env, h_env: Operating conditions
            parameter_ranges: Dict with keys like "T_gas", "thickness_0", etc.
                            Values are (min, max) tuples
            n_points: Number of points per parameter sweep
            verbose: Print details
            
        Returns:
            Dictionary with sensitivity data
        """
        results = {}
        
        # Baseline solution
        baseline = self.solve_steady_state_iterative(
            T_gas, h_gas, T_env, h_env, verbose=False
        )
        T_outer_baseline = baseline["T_outer"]
        
        for param_name, (val_min, val_max) in parameter_ranges.items():
            values = np.linspace(val_min, val_max, n_points)
            T_outer_vals = []
            
            for val in values:
                # Temporarily modify parameter
                if param_name == "T_gas":
                    sol = self.solve_steady_state_iterative(
                        val, h_gas, T_env, h_env, verbose=False
                    )
                elif param_name == "h_gas":
                    sol = self.solve_steady_state_iterative(
                        T_gas, val, T_env, h_env, verbose=False
                    )
                elif param_name == "thickness_0":
                    self.materials[0].thickness = val
                    sol = self.solve_steady_state_iterative(
                        T_gas, h_gas, T_env, h_env, verbose=False
                    )
                    self.materials[0].thickness = parameter_ranges[param_name][0] + \
                        (parameter_ranges[param_name][1] - parameter_ranges[param_name][0]) / 2
                else:
                    continue
                
                T_outer_vals.append(sol["T_outer"])
            
            sensitivity = (max(T_outer_vals) - min(T_outer_vals)) / abs(T_outer_baseline)
            
            results[param_name] = {
                "values": values,
                "T_outer": T_outer_vals,
                "sensitivity": sensitivity,
            }
        
        return results


# ============================================================================
# Visualization and Reporting
# ============================================================================

def plot_temperature_profile(analyzer: ThermalMultilayerAnalyzer, 
                            solution: Dict, save_path: str = None):
    """Plot temperature profile through multilayer wall."""
    if solution is None or "T_layers" not in solution:
        print("No solution available for plotting")
        return
    
    T_layers = solution["T_layers"]
    distances = [0.0]
    
    for mat in analyzer.materials:
        distances.append(distances[-1] + mat.thickness * 1000)  # Convert to mm
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Temperature profile
    ax1.plot(distances, T_layers, 'b-o', linewidth=2.5, markersize=8, label="Temperature")
    ax1.axhline(y=solution.get("T_outer", 0), color='r', linestyle='--', label="T_outer")
    ax1.set_xlabel("Distance from inner surface [mm]", fontsize=12)
    ax1.set_ylabel("Temperature [K]", fontsize=12)
    ax1.set_title("Temperature Profile Through Multilayer Wall", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Thermal resistance breakdown
    R_data = {
        "Convection\n(Gas)": solution.get("R_conv_gas", 0),
    }
    
    for i, R_cond in enumerate(solution.get("R_cond_layers", [])):
        mat = analyzer.materials[i]
        R_data[f"Layer {i+1}\n({mat.name})"] = R_cond
    
    R_data["Convection\n(Environment)"] = solution.get("R_conv_env", 0)
    if solution.get("R_rad_env", float('inf')) != float('inf'):
        R_data["Radiation"] = min(solution.get("R_rad_env", float('inf')), 10)
    
    labels = list(R_data.keys())
    values = list(R_data.values())
    colors = plt.cm.Spectral(np.linspace(0, 1, len(labels)))
    
    bars = ax2.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel("Thermal Resistance [K/W]", fontsize=12)
    ax2.set_title("Thermal Resistance Breakdown", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if height > 0.01:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    
    plt.show()


def print_solution_summary(analyzer: ThermalMultilayerAnalyzer, solution: Dict):
    """Print comprehensive solution summary."""
    print("\n" + "="*80)
    print(" THERMAL MULTILAYER WALL ANALYSIS - STEADY-STATE SOLUTION")
    print("="*80)
    
    print(f"\n{'BOUNDARY CONDITIONS':^80}")
    print("-"*80)
    print(f"  Gas side:        T_gas = {solution.get('T_gas', 1000):.1f} K, "
          f"h_gas = {solution.get('h_gas', 50):.1f} W/m²/K")
    print(f"  Environment:     T_env = {solution.get('T_env', 300):.1f} K, "
          f"h_env = {solution.get('h_env', 10):.1f} W/m²/K")
    print(f"  Convergence:     {solution.get('iterations', 0)} iterations, "
          f"residual = {solution.get('residual', 0):.2e} K")
    
    print(f"\n{'HEAT TRANSFER RESULTS':^80}")
    print("-"*80)
    q = solution.get("heat_flux", 0)
    print(f"  Heat flux:       q = {q:.2f} W/m²")
    print(f"  Inner surface:   T_inner = {solution.get('T_inner', 0):.2f} K")
    print(f"  Outer surface:   T_outer = {solution.get('T_outer', 0):.2f} K")
    print(f"  ΔT (outer-env):  {solution.get('T_outer', 0) - 300:.2f} K")
    
    print(f"\n{'LAYER TEMPERATURES':^80}")
    print("-"*80)
    print(f"  {'Layer':<20} {'From [K]':<15} {'To [K]':<15} {'ΔT [K]':<15}")
    print("-"*80)
    
    T_layers = solution.get("T_layers", [])
    for i, mat in enumerate(analyzer.materials):
        T_in = T_layers[i] if i < len(T_layers) else 0
        T_out = T_layers[i+1] if i+1 < len(T_layers) else 0
        print(f"  {mat.name:<20} {T_in:>14.2f} {T_out:>14.2f} {T_in - T_out:>14.2f}")
    
    print(f"\n{'THERMAL RESISTANCE':^80}")
    print("-"*80)
    print(f"  {'Component':<30} {'Resistance [K/W]':<20} {'Share [%]':<15}")
    print("-"*80)
    
    R_gas = solution.get("R_conv_gas", 0)
    R_cond = solution.get("R_total_cond", 0)
    R_env = solution.get("R_conv_env", 0)
    R_rad = solution.get("R_rad_env", 0)
    
    R_total = R_gas + R_cond + R_env + (R_rad if R_rad != float('inf') else 0)
    if R_total > 0:
        print(f"  Gas convection   {R_gas:>19.4f} {100*R_gas/R_total:>14.1f}%")
        print(f"  Conduction       {R_cond:>19.4f} {100*R_cond/R_total:>14.1f}%")
        print(f"  Env. convection  {R_env:>19.4f} {100*R_env/R_total:>14.1f}%")
        if R_rad != float('inf'):
            print(f"  Radiation        {R_rad:>19.4f} {100*R_rad/R_total:>14.1f}%")
        print("-"*80)
        print(f"  TOTAL            {R_total:>19.4f} {'100.0':>14}%")
    
    print("\n" + "="*80)


# ============================================================================
# Example Application: High-Temperature Furnace Wall
# ============================================================================

def example_furnace_analysis():
    """
    Example: Analyze a multi-layer furnace wall with realistic parameters.
    
    Setup:
    ------
    Furnace: 1000°C flue gas
    Wall structure (inner to outer):
      1. High-temp refractory (100 mm)
      2. Insulation (150 mm)
      3. Steel shell (5 mm)
    
    Environment: 25°C ambient, natural convection
    """
    
    print("\n" + "="*80)
    print(" EXAMPLE: HIGH-TEMPERATURE FURNACE WALL ANALYSIS")
    print("="*80)
    
    # Define materials
    refractory = ThermalMaterial(
        name="High-Temp Refractory",
        thickness=0.1,  # 100 mm
        k0=0.5,         # W/m/K @ 300K
        rho=2400,
        cp=1200,
        a=5e-4,         # High-temp material (k increases with T)
        b=0.0
    )
    
    insulation = ThermalMaterial(
        name="Fiber Insulation",
        thickness=0.15, # 150 mm
        k0=0.08,
        rho=200,
        cp=1000,
        a=3e-4,
        b=0.0
    )
    
    steel_shell = ThermalMaterial(
        name="Steel Shell",
        thickness=0.005, # 5 mm
        k0=50.0,
        rho=7850,
        cp=490,
        a=0.0,
        b=0.0
    )
    
    materials = [refractory, insulation, steel_shell]
    
    # Create analyzer
    analyzer = ThermalMultilayerAnalyzer(
        materials=materials,
        geometry=ThermalConfig.GEOMETRY_PLANAR,
        convergence_tol=1e-4,
        max_iterations=100
    )
    
    # Operating conditions
    T_gas = 1273.15  # 1000°C
    T_env = 298.15   # 25°C
    
    # Compute convection coefficients
    # Gas side: high velocity combustion gases
    Re_gas = 10000  # High Reynolds number
    Pr_gas = AirProperties.prandtl_number(T_gas)
    Nu_gas = ConvectionModel.churchill_bernstein(Re_gas, Pr_gas)
    k_gas = AirProperties.thermal_conductivity(T_gas)
    h_gas = ConvectionModel.heat_transfer_coefficient(Nu_gas, k_gas, 0.1)
    
    # Environment side: natural convection on vertical surface
    Re_env = 1e5  # Moderate Reynolds number
    Pr_env = AirProperties.prandtl_number(T_env)
    Nu_env = ConvectionModel.churchill_bernstein(Re_env, Pr_env)
    k_env = AirProperties.thermal_conductivity(T_env)
    h_env = ConvectionModel.heat_transfer_coefficient(Nu_env, k_env, 1.0)
    
    print(f"\nConvection Analysis:")
    print(f"  Gas side:  Re={Re_gas:.0e}, Pr={Pr_gas:.3f}, Nu={Nu_gas:.1f}, h={h_gas:.1f} W/m²/K")
    print(f"  Env side:  Re={Re_env:.0e}, Pr={Pr_env:.3f}, Nu={Nu_env:.1f}, h={h_env:.1f} W/m²/K")
    
    # Solve
    solution = analyzer.solve_steady_state_iterative(
        T_gas=T_gas,
        h_gas=h_gas,
        T_env=T_env,
        h_env=h_env,
        emissivity_outer=0.85,
        T_sky=T_env,
        verbose=True
    )
    
    solution["T_gas"] = T_gas
    solution["T_env"] = T_env
    
    print_solution_summary(analyzer, solution)
    
    # Plot
    plot_temperature_profile(analyzer, solution, save_path="furnace_temperature_profile.png")
    
    # Sensitivity analysis
    print(f"\n{'SENSITIVITY ANALYSIS':^80}")
    print("-"*80)
    
    sens_params = {
        "T_gas": (1173.15, 1373.15),  # ±100 K
        "h_gas": (50, 200),
        "thickness_0": (0.05, 0.15),
    }
    
    sensitivity = analyzer.compute_sensitivity(
        T_gas, h_gas, T_env, h_env,
        parameter_ranges=sens_params,
        n_points=5,
        verbose=True
    )
    
    for param, data in sensitivity.items():
        dT_outer = max(data["T_outer"]) - min(data["T_outer"])
        print(f"  {param:20}: ΔT_outer = ±{dT_outer/2:.1f} K (sensitivity = {data['sensitivity']:.2%})")
    
    return analyzer, solution


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Set up matplotlib for Chinese characters if needed
    rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
    
    # Run example
    analyzer, solution = example_furnace_analysis()
    
    print("\n✓ Analysis complete!")
