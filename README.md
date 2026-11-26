# Dual-Sensor Observer-Based Control and Residual Safety Monitoring for a Non-Contact Radiometric Thermal Therapy Blanket

Reproducible Python simulation suite for a coupled two-node thermal therapy blanket model comparing hysteresis relay control, SIMC-tuned PI control, and an augmented Kalman observer + LQR controller with dual residual fault detection.

## Repository Structure

* **`plant2.py`**: Defines the continuous-time two-node RC thermal network (heating plane $C_h = 2600\,\mathrm{J/K}$, skin-side surface $C_s = 1500\,\mathrm{J/K}$, textile resistance $R_{hs} = 0.0113\,\mathrm{K/W}$, ambient resistance $R_{so} = 0.0338\,\mathrm{K/W}$, $P_{\max} = 550\,\mathrm{W}$), exact zero-order-hold discretization ($T_s = 0.25\,\mathrm{s}$), and first-order-plus-dead-time (FOPDT) half-rule reduction.
* **`ctrl.py`**: Implements the four controllers and closed-loop simulator:
  * `Relay`: Hysteresis bang-bang controllers on the buried NTC (`Relay-NTC`) and non-contact IR surface sensor (`Relay-IR`).
  * `PI`: SIMC-tuned proportional-integral controller with back-calculation anti-windup and $4\,\mathrm{s}$ error-carry zero-cross PWM (`PI-IR`).
  * `Obs`: 3-state augmented Kalman observer (`[Th, Ts, Q]`), discrete LQR state feedback with conditional integral action, plane-temperature governor, normalized innovation fault detector ($\nu_k$), and $60\,\mathrm{s}$ sliding-window plane energy-balance residual detector ($r_g$).
* **`final_study.py`**: Runs the 200-trial Monte Carlo robustness ablation ($\pm 15\%$ parametric mismatch, sensor bias/noise, cold-load step), fault-injection benchmarks (abrupt step faults and slow sensor drift), and generates the 300 DPI comparison figure (`sim_results.png`).

## Requirements

* Python 3.9+
* `numpy`
* `scipy`
* `matplotlib`

Install dependencies via:

```bash
pip install numpy scipy matplotlib
```

## Reproducing the Results

1. **Verify open-loop plant poles, FOPDT parameters, and step rise time:**

```bash
python3 plant2.py
```

2. **Run the 200-trial Monte Carlo ablation, fault-injection statistics, and generate `sim_results.png`:**

```bash
python3 final_study.py
```

## License

This code is released under the MIT License for academic verification and reproducibility.
