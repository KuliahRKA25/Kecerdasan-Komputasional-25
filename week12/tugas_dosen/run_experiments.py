"""
Script Eksperimen PSO Komprehensif
Mata Kuliah: Kecerdasan Komputasional
Penulis: Benedictus Ryu Gunawan (NRP: 5054251001)
Institut Teknologi Sepuluh Nopember (ITS)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Set aesthetic styling for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Definisi Fungsi Benchmark
# ---------------------------------------------------------
def sphere(x):
    """
    Fungsi Sphere: f(x) = sum(x_i^2)
    Global minimum: f(x*) = 0 pada x* = (0, ..., 0)
    Karakteristik: Unimodal, konveks, kontinu, mulus.
    """
    x = np.asarray(x)
    return np.sum(x ** 2, axis=-1)

def rastrigin(x):
    """
    Fungsi Rastrigin: f(x) = 10*D + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum: f(x*) = 0 pada x* = (0, ..., 0)
    Karakteristik: Multimodal dengan modulasi sinusoidal,
                   memiliki ribuan minimum lokal dalam [-5.12, 5.12].
    """
    x = np.asarray(x)
    D = x.shape[-1]
    return 10.0 * D + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1)

# ---------------------------------------------------------
# 2. Implementasi PSO gbest dari Nol
# ---------------------------------------------------------
class ParticleSwarmOptimizer:
    """
    Implementasi gbest Particle Swarm Optimization (PSO) murni dengan NumPy.
    Dilengkapi fitur:
      - Velocity clamping: v in [-v_max, v_max]
      - Boundary handling: Position clipping + reset velocity pada batas
      - Tiga varian bobot:
          1. 'constant_w' (w = 0.7, c1 = c2 = 1.49618)
          2. 'linear_w'   (w menurun linier 0.9 -> 0.4, c1 = c2 = 2.0)
          3. 'constriction' (chi = 0.7298, phi1 = phi2 = 2.05)
    """
    def __init__(self, obj_func, dim=10, bounds=(-5.12, 5.12), num_particles=30,
                 max_iter=200, variant='constant_w', w_val=0.7,
                 w_start=0.9, w_end=0.4, c1=2.0, c2=2.0,
                 chi=0.7298, phi1=2.05, phi2=2.05,
                 v_max_ratio=0.2, seed=None):
        self.obj_func = obj_func
        self.dim = dim
        self.lower_bound, self.upper_bound = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.variant = variant
        
        # Parameter bobot inersia & kognitif/sosial
        self.w_val = w_val
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.chi = chi
        self.phi1 = phi1
        self.phi2 = phi2
        
        # Velocity clamping
        self.bound_range = self.upper_bound - self.lower_bound
        self.v_max = v_max_ratio * self.bound_range
        
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def optimize(self):
        # 1. Inisialisasi Posisi dan Kecepatan Partikel
        # Posisi diinisialisasi secara seragam dalam rentang [lower_bound, upper_bound]
        X = self.rng.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        # Kecepatan diinisialisasi dalam [-v_max, v_max]
        V = self.rng.uniform(-self.v_max, self.v_max, (self.num_particles, self.dim))
        
        # 2. Evaluasi Fitness Awal
        fitness = self.obj_func(X)
        
        # Inisialisasi pbest (personal best)
        pbest_X = np.copy(X)
        pbest_fit = np.copy(fitness)
        
        # Inisialisasi gbest (global best)
        best_idx = np.argmin(pbest_fit)
        gbest_X = np.copy(pbest_X[best_idx])
        gbest_fit = float(pbest_fit[best_idx])
        
        history_gbest_fit = [gbest_fit]
        
        # 3. Iterasi Optimasi
        for t in range(1, self.max_iter + 1):
            # Bilangan acak seragam r1, r2 ~ U(0, 1) untuk setiap partikel dan dimensi
            r1 = self.rng.uniform(0.0, 1.0, (self.num_particles, self.dim))
            r2 = self.rng.uniform(0.0, 1.0, (self.num_particles, self.dim))
            
            # Update Kecepatan berdasarkan Varian
            if self.variant == 'constant_w':
                # w tetap = 0.7, c1 = c2 = 1.49618 (atau parameter konstan c1, c2)
                cognitive = self.c1 * r1 * (pbest_X - X)
                social = self.c2 * r2 * (gbest_X - X)
                V = self.w_val * V + cognitive + social
                
            elif self.variant == 'linear_w':
                # w menurun linier dari w_start (0.9) ke w_end (0.4)
                w_t = self.w_start - (t / self.max_iter) * (self.w_start - self.w_end)
                cognitive = self.c1 * r1 * (pbest_X - X)
                social = self.c2 * r2 * (gbest_X - X)
                V = w_t * V + cognitive + social
                
            elif self.variant == 'constriction':
                # Clerc & Kennedy (2002): constriction factor chi = 0.7298
                # v(t+1) = chi * [v(t) + phi1*r1*(pbest - x) + phi2*r2*(gbest - x)]
                cognitive = self.phi1 * r1 * (pbest_X - X)
                social = self.phi2 * r2 * (gbest_X - X)
                V = self.chi * (V + cognitive + social)
            else:
                raise ValueError(f"Unknown variant: {self.variant}")
                
            # Velocity Clamping: membatasi kecepatan dalam [-v_max, v_max]
            V = np.clip(V, -self.v_max, self.v_max)
            
            # Update Posisi
            X_new = X + V
            
            # Penanganan Batas (Boundary Handling):
            # Posisi yang melanggar batas di-clamp ke batas terdekat,
            # dan kecepatan pada dimensi tersebut di-reset (v = 0) agar partikel tidak terjebak terus menekan batas.
            out_lower = X_new < self.lower_bound
            out_upper = X_new > self.upper_bound
            
            X_new = np.clip(X_new, self.lower_bound, self.upper_bound)
            # Reset kecepatan bila membentur batas
            V[out_lower | out_upper] = 0.0
            
            X = X_new
            
            # Evaluasi fitness partikel baru
            fitness = self.obj_func(X)
            
            # Update pbest
            improved_mask = fitness < pbest_fit
            pbest_X[improved_mask] = X[improved_mask]
            pbest_fit[improved_mask] = fitness[improved_mask]
            
            # Update gbest
            current_best_idx = np.argmin(pbest_fit)
            if pbest_fit[current_best_idx] < gbest_fit:
                gbest_fit = float(pbest_fit[current_best_idx])
                gbest_X = np.copy(pbest_X[current_best_idx])
                
            history_gbest_fit.append(gbest_fit)
            
        return {
            'gbest_X': gbest_X,
            'gbest_fit': gbest_fit,
            'history': np.array(history_gbest_fit)
        }

# ---------------------------------------------------------
# 3. BONUS: Binary PSO (BPSO) untuk Seleksi Fitur
# ---------------------------------------------------------
class BinaryPSOFeatureSelection:
    """
    Implementasi Binary Particle Swarm Optimization (BPSO) dari nol
    untuk seleksi fitur pada dataset UCI (Breast Cancer Wisconsin).
    
    Karakteristik:
      - Posisi bit biner: x_id in {0, 1}
      - Kecepatan kontinu: v_id in [-v_max, v_max]
      - Transfer Function Sigmoid: S(v_id) = 1 / (1 + exp(-v_id))
      - Pembaruan Posisi: x_id = 1 jika rand() < S(v_id) else 0
      - Fungsi Objektif (Minimisasi):
          Fitness = alpha * (1 - Accuracy) + (1 - alpha) * (|Fitur Terpilih| / Total Fitur)
    """
    def __init__(self, X_data, y_data, num_particles=25, max_iter=60,
                 w=0.7, c1=1.5, c2=1.5, v_max=6.0, alpha=0.95, seed=42):
        self.X_data = X_data
        self.y_data = y_data
        self.num_samples, self.num_features = X_data.shape
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
    def _sigmoid(self, v):
        # Clip untuk mencegah numeric overflow
        v_clipped = np.clip(v, -15.0, 15.0)
        return 1.0 / (1.0 + np.exp(-v_clipped))
        
    def _evaluate_particle(self, bitstring):
        selected_indices = np.where(bitstring == 1)[0]
        num_selected = len(selected_indices)
        
        # Jika tidak ada fitur terpilih sama sekali, berikan penalti maksimal
        if num_selected == 0:
            return 1.0, 0.0, 0
            
        X_subset = self.X_data[:, selected_indices]
        clf = KNeighborsClassifier(n_neighbors=5)
        # 5-fold Cross Validation Accuracy
        scores = cross_val_score(clf, X_subset, self.y_data, cv=self.cv, scoring='accuracy')
        mean_acc = np.mean(scores)
        error_rate = 1.0 - mean_acc
        
        # Multi-objective fitness function
        ratio_feat = num_selected / self.num_features
        fitness = self.alpha * error_rate + (1.0 - self.alpha) * ratio_feat
        return fitness, mean_acc, num_selected

    def optimize(self):
        # 1. Inisialisasi Posisi Biner & Kecepatan Kontinu
        # Probabilitas awal 50% setiap fitur terpilih
        X = self.rng.integers(0, 2, (self.num_particles, self.num_features))
        # Pastikan tidak ada partikel dengan 0 fitur terpilih
        for i in range(self.num_particles):
            if np.sum(X[i]) == 0:
                X[i, self.rng.integers(0, self.num_features)] = 1
                
        V = self.rng.uniform(-self.v_max, self.v_max, (self.num_particles, self.num_features))
        
        # 2. Evaluasi Awal
        pbest_X = np.copy(X)
        pbest_fit = np.zeros(self.num_particles)
        pbest_acc = np.zeros(self.num_particles)
        pbest_nfeat = np.zeros(self.num_particles, dtype=int)
        
        for i in range(self.num_particles):
            fit, acc, nf = self._evaluate_particle(X[i])
            pbest_fit[i] = fit
            pbest_acc[i] = acc
            pbest_nfeat[i] = nf
            
        best_idx = np.argmin(pbest_fit)
        gbest_X = np.copy(pbest_X[best_idx])
        gbest_fit = float(pbest_fit[best_idx])
        gbest_acc = float(pbest_acc[best_idx])
        gbest_nfeat = int(pbest_nfeat[best_idx])
        
        history_fitness = [gbest_fit]
        history_acc = [gbest_acc]
        history_nfeat = [gbest_nfeat]
        
        # 3. Iterasi Optimasi BPSO
        for t in range(1, self.max_iter + 1):
            r1 = self.rng.uniform(0.0, 1.0, (self.num_particles, self.num_features))
            r2 = self.rng.uniform(0.0, 1.0, (self.num_particles, self.num_features))
            
            # Pembaruan kecepatan kontinu BPSO
            cognitive = self.c1 * r1 * (pbest_X - X)
            social = self.c2 * r2 * (gbest_X - X)
            V = self.w * V + cognitive + social
            V = np.clip(V, -self.v_max, self.v_max)
            
            # Sigmoid transfer function
            prob = self._sigmoid(V)
            rand_matrix = self.rng.uniform(0.0, 1.0, (self.num_particles, self.num_features))
            
            # Pembaruan posisi biner
            X_new = (rand_matrix < prob).astype(int)
            
            # Repairing mechanism: jika ada partikel yang memilih 0 fitur, aktifkan setidaknya 1 fitur acak
            for i in range(self.num_particles):
                if np.sum(X_new[i]) == 0:
                    best_prob_idx = np.argmax(prob[i])
                    X_new[i, best_prob_idx] = 1
                    
            X = X_new
            
            # Evaluasi partikel baru
            for i in range(self.num_particles):
                fit, acc, nf = self._evaluate_particle(X[i])
                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_X[i] = np.copy(X[i])
                    pbest_acc[i] = acc
                    pbest_nfeat[i] = nf
                    
            current_best_idx = np.argmin(pbest_fit)
            if pbest_fit[current_best_idx] < gbest_fit:
                gbest_fit = float(pbest_fit[current_best_idx])
                gbest_X = np.copy(pbest_X[current_best_idx])
                gbest_acc = float(pbest_acc[current_best_idx])
                gbest_nfeat = int(pbest_nfeat[current_best_idx])
                
            history_fitness.append(gbest_fit)
            history_acc.append(gbest_acc)
            history_nfeat.append(gbest_nfeat)
            
        return {
            'gbest_features': gbest_X,
            'gbest_fit': gbest_fit,
            'gbest_acc': gbest_acc,
            'gbest_nfeat': gbest_nfeat,
            'history_fitness': np.array(history_fitness),
            'history_acc': np.array(history_acc),
            'history_nfeat': np.array(history_nfeat)
        }

# ---------------------------------------------------------
# 4. Prosedur Eksperimen Benchmark (Sphere & Rastrigin)
# ---------------------------------------------------------
def run_benchmark_experiments():
    print("=== Memulai Eksperimen Benchmark PSO (Sphere & Rastrigin) ===")
    num_runs = 30
    num_particles = 30
    max_iter = 200
    dim = 10
    bounds = (-5.12, 5.12)
    
    variants = [
        ('constant_w', 'w Tetap (0.7)', {'variant': 'constant_w', 'w_val': 0.7, 'c1': 1.49618, 'c2': 1.49618}),
        ('linear_w', 'w Menurun (0.9 -> 0.4)', {'variant': 'linear_w', 'w_start': 0.9, 'w_end': 0.4, 'c1': 2.0, 'c2': 2.0}),
        ('constriction', 'Constriction (chi=0.7298)', {'variant': 'constriction', 'chi': 0.7298, 'phi1': 2.05, 'phi2': 2.05})
    ]
    
    functions = [
        ('Sphere', sphere),
        ('Rastrigin', rastrigin)
    ]
    
    results = {}
    
    for func_name, func in functions:
        results[func_name] = {}
        for var_key, var_label, params in variants:
            print(f"Menjalankan {func_name} | {var_label} ({num_runs} runs)...")
            runs_best_fit = []
            runs_history = []
            
            for run_i in range(num_runs):
                seed = 1000 + run_i * 17
                optimizer = ParticleSwarmOptimizer(
                    obj_func=func,
                    dim=dim,
                    bounds=bounds,
                    num_particles=num_particles,
                    max_iter=max_iter,
                    seed=seed,
                    **params
                )
                res = optimizer.optimize()
                runs_best_fit.append(res['gbest_fit'])
                runs_history.append(res['history'])
                
            runs_best_fit = np.array(runs_best_fit)
            runs_history = np.array(runs_history)  # shape: (num_runs, max_iter + 1)
            
            results[func_name][var_key] = {
                'label': var_label,
                'best_fits': runs_best_fit.tolist(),
                'mean_fit': float(np.mean(runs_best_fit)),
                'std_fit': float(np.std(runs_best_fit)),
                'min_fit': float(np.min(runs_best_fit)),
                'median_fit': float(np.median(runs_best_fit)),
                'max_fit': float(np.max(runs_best_fit)),
                'history_mean': np.mean(runs_history, axis=0).tolist(),
                'history_std': np.std(runs_history, axis=0).tolist(),
                'history_min': np.min(runs_history, axis=0).tolist(),
                'history_max': np.max(runs_history, axis=0).tolist()
            }
            print(f"  -> Mean +- Std: {np.mean(runs_best_fit):.4e} +- {np.std(runs_best_fit):.4e} | Best: {np.min(runs_best_fit):.4e}")
            
    return results

# ---------------------------------------------------------
# 5. Visualisasi Benchmark & Plotting
# ---------------------------------------------------------
def generate_benchmark_figures(results):
    print("=== Mengenerate Gambar Visualisasi Benchmark ===")
    
    # 1. 2D / 3D Landscape Visualization
    fig = plt.figure(figsize=(12, 5))
    x = np.linspace(-5.12, 5.12, 200)
    y = np.linspace(-5.12, 5.12, 200)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # Sphere 2D contour
    Z_sphere = X_grid**2 + Y_grid**2
    ax1 = fig.add_subplot(1, 2, 1)
    cp1 = ax1.contourf(X_grid, Y_grid, Z_sphere, levels=40, cmap='viridis')
    fig.colorbar(cp1, ax=ax1, label='Nilai Fitness f(x)')
    ax1.plot(0, 0, 'r*', markersize=14, label='Global Optimum (0, 0)')
    ax1.set_title("Kontur Lanskap Fungsi Sphere (Unimodal)")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend(loc='upper right')
    
    # Rastrigin 2D contour
    Z_rast = 20.0 + (X_grid**2 - 10.0*np.cos(2*np.pi*X_grid)) + (Y_grid**2 - 10.0*np.cos(2*np.pi*Y_grid))
    ax2 = fig.add_subplot(1, 2, 2)
    cp2 = ax2.contourf(X_grid, Y_grid, Z_rast, levels=40, cmap='plasma')
    fig.colorbar(cp2, ax=ax2, label='Nilai Fitness f(x)')
    ax2.plot(0, 0, 'w*', markersize=14, label='Global Optimum (0, 0)')
    ax2.set_title("Kontur Lanskap Fungsi Rastrigin (Multimodal)")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    landscape_path = os.path.join(FIG_DIR, "lanskap_fungsi_uji.png")
    plt.savefig(landscape_path, dpi=300)
    plt.close()
    print(f"Disimpan: {landscape_path}")
    
    # 2. Kurva Konvergensi Perbandingan (Sphere & Rastrigin)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {
        'constant_w': '#1f77b4',   # Biru
        'linear_w': '#2ca02c',     # Hijau
        'constriction': '#d62728'  # Merah
    }
    line_styles = {
        'constant_w': '--',
        'linear_w': '-',
        'constriction': '-.'
    }
    
    for ax_idx, (func_name, ax) in enumerate(zip(['Sphere', 'Rastrigin'], axes)):
        for var_key, color in colors.items():
            var_data = results[func_name][var_key]
            mean_hist = np.array(var_data['history_mean'])
            std_hist = np.array(var_data['history_std'])
            label = var_data['label']
            iters = np.arange(len(mean_hist))
            
            ax.plot(iters, mean_hist, label=label, color=color,
                    linestyle=line_styles[var_key], linewidth=2.0)
            # Area pita standar deviasi (clipped agar tidak negatif untuk log-scale)
            lower = np.maximum(mean_hist - std_hist, 1e-15 if func_name == 'Sphere' else 1e-4)
            upper = mean_hist + std_hist
            ax.fill_between(iters, lower, upper, color=color, alpha=0.15)
            
        ax.set_title(f"Kurva Konvergensi Rata-rata 30 Run: {func_name} (D=10)", fontweight='bold')
        ax.set_xlabel("Iterasi")
        ax.set_ylabel("Fitness Terbaik (Skala Logaritmik)")
        ax.set_yscale('log')
        ax.legend(frameon=True, loc='upper right')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
    plt.tight_layout()
    conv_path = os.path.join(FIG_DIR, "kurva_konvergensi_pso.png")
    plt.savefig(conv_path, dpi=300)
    plt.close()
    print(f"Disimpan: {conv_path}")
    
    # 3. Boxplot Distribusi Nilai Fitness Akhir
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, (func_name, ax) in enumerate(zip(['Sphere', 'Rastrigin'], axes)):
        data_to_plot = [results[func_name][var_key]['best_fits'] for var_key in colors.keys()]
        labels = [results[func_name][var_key]['label'] for var_key in colors.keys()]
        
        box = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=labels,
                         medianprops=dict(color='black', linewidth=1.5),
                         flierprops=dict(marker='o', markersize=5, alpha=0.6))
        
        palette = ['#aec7e8', '#98df8a', '#ff9896']
        for patch, col in zip(box['boxes'], palette):
            patch.set_facecolor(col)
            
        ax.set_title(f"Distribusi Fitness Terbaik 30 Run: {func_name}", fontweight='bold')
        ax.set_ylabel("Nilai Fitness Akhir (Log Scale)")
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
    plt.tight_layout()
    box_path = os.path.join(FIG_DIR, "boxplot_fitness_pso.png")
    plt.savefig(box_path, dpi=300)
    plt.close()
    print(f"Disimpan: {box_path}")

# ---------------------------------------------------------
# 6. Prosedur Eksperimen BONUS (BPSO Feature Selection)
# ---------------------------------------------------------
def run_bpso_bonus_experiments():
    print("\n=== Memulai Eksperimen Bonus Binary PSO (Seleksi Fitur UCI) ===")
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = cancer.feature_names
    
    # Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Baseline dengan SEMUA 30 fitur menggunakan k-NN (k=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_base = KNeighborsClassifier(n_neighbors=5)
    baseline_scores = cross_val_score(clf_base, X_scaled, y, cv=cv, scoring='accuracy')
    baseline_acc = float(np.mean(baseline_scores))
    baseline_std = float(np.std(baseline_scores))
    
    print(f"Baseline (Semua 30 Fitur): Akurasi 5-Fold CV = {baseline_acc*100:.2f}% +- {baseline_std*100:.2f}%")
    
    # Jalankan BPSO
    bpso = BinaryPSOFeatureSelection(
        X_data=X_scaled,
        y_data=y,
        num_particles=25,
        max_iter=60,
        w=0.7,
        c1=1.5,
        c2=1.5,
        v_max=6.0,
        alpha=0.95,
        seed=42
    )
    bpso_res = bpso.optimize()
    
    selected_mask = bpso_res['gbest_features']
    selected_feat_indices = np.where(selected_mask == 1)[0]
    selected_feat_names = [feature_names[i] for i in selected_feat_indices]
    
    print(f"BPSO Selesai!")
    print(f"Fitur Terpilih: {len(selected_feat_indices)} / {len(feature_names)} ({len(selected_feat_indices)/len(feature_names)*100:.1f}%)")
    print(f"Akurasi BPSO: {bpso_res['gbest_acc']*100:.2f}% (Peningkatan: {(bpso_res['gbest_acc'] - baseline_acc)*100:+.2f}%)")
    print(f"Nama Fitur Terpilih: {selected_feat_names}")
    
    # Simpan Visualisasi BPSO
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Kurva konvergensi BPSO (Fitness & Akurasi)
    iters = np.arange(len(bpso_res['history_fitness']))
    ax1 = axes[0]
    ax1.plot(iters, bpso_res['history_fitness'], 'b-', linewidth=2, label='Nilai Fitness (Minimisasi)')
    ax1.set_xlabel("Iterasi BPSO")
    ax1.set_ylabel("Fitness Objektif", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(iters, bpso_res['history_acc'] * 100, 'g--', linewidth=2, label='Akurasi CV (%)')
    ax1_twin.set_ylabel("Akurasi 5-Fold CV (%)", color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')
    ax1.set_title("Kurva Konvergensi Binary PSO (Seleksi Fitur)", fontweight='bold')
    
    # Perbandingan Fitur dan Akurasi
    ax2 = axes[1]
    categories = ['Semua Fitur\n(Baseline)', 'Fitur Terpilih\n(BPSO)']
    features_count = [len(feature_names), len(selected_feat_indices)]
    accuracies = [baseline_acc * 100, bpso_res['gbest_acc'] * 100]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax2.bar(x_pos - width/2, features_count, width, label='Jumlah Fitur', color='#4a90e2')
    ax2.set_ylabel('Jumlah Fitur (dari 30)', color='#4a90e2')
    ax2.tick_params(axis='y', labelcolor='#4a90e2')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 35)
    
    ax2_twin = ax2.twinx()
    rects2 = ax2_twin.bar(x_pos + width/2, accuracies, width, label='Akurasi (%)', color='#50e3c2')
    ax2_twin.set_ylabel('Akurasi Klasifikasi (%)', color='#2ca02c')
    ax2_twin.tick_params(axis='y', labelcolor='#2ca02c')
    ax2_twin.set_ylim(90, 100)
    
    # Nilai di atas bar
    for rect in rects1:
        h = rect.get_height()
        ax2.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax2_twin.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                          xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
        
    ax2.set_title("Perbandingan Reduksi Fitur & Akurasi", fontweight='bold')
    plt.tight_layout()
    bpso_fig_path = os.path.join(FIG_DIR, "bpso_seleksi_fitur.png")
    plt.savefig(bpso_fig_path, dpi=300)
    plt.close()
    print(f"Disimpan: {bpso_fig_path}")
    
    bonus_data = {
        'dataset_name': 'UCI Breast Cancer Wisconsin (Diagnostic)',
        'num_samples': int(cancer.data.shape[0]),
        'num_features_orig': int(cancer.data.shape[1]),
        'baseline_accuracy': baseline_acc,
        'baseline_accuracy_std': baseline_std,
        'num_features_selected': int(len(selected_feat_indices)),
        'selected_features_names': selected_feat_names,
        'bpso_accuracy': float(bpso_res['gbest_acc']),
        'accuracy_gain': float(bpso_res['gbest_acc'] - baseline_acc),
        'feature_reduction_percent': float((1 - len(selected_feat_indices)/len(feature_names)) * 100),
        'history_fitness': bpso_res['history_fitness'].tolist(),
        'history_acc': bpso_res['history_acc'].tolist(),
        'history_nfeat': bpso_res['history_nfeat'].tolist()
    }
    return bonus_data

if __name__ == '__main__':
    bench_results = run_benchmark_experiments()
    generate_benchmark_figures(bench_results)
    bonus_results = run_bpso_bonus_experiments()
    
    all_summary = {
        'benchmark': bench_results,
        'bonus': bonus_results
    }
    
    # Save JSON for fast reading and report compilation
    json_path = os.path.join(OUTPUT_DIR, "hasil_eksperimen.json")
    with open(json_path, 'w') as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nSemua data eksperimen berhasil disimpan di: {json_path}")
