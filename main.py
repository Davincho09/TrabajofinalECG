import numpy as np
import matplotlib.pyplot as plt
from deap import tools  
from ecg_utils import load_ecg, bandpass_filter, detect_r_peaks, extract_beats
from neuroevolution import evolve_autoencoder, SimpleAutoencoder


ecg = load_ecg("ECG1kHz.csv")
filtered = bandpass_filter(ecg)
peaks = detect_r_peaks(filtered)
beats = extract_beats(filtered, peaks)

beats = (beats - np.min(beats)) / (np.max(beats) - np.min(beats))

pop, logbook = evolve_autoencoder(beats, latent_dim=5, pop_size=40, ngen=30)

print("Evolución completada.")


best_individuals = tools.selBest(pop, 10)
input_dim = beats.shape[1]
latent_dim = 5  
mse_list, complexity_list = [], []
for ind in best_individuals:
    ae = SimpleAutoencoder(input_dim, latent_dim, np.array(ind))
    mse_val = ae.mse(beats)
    comp_val = np.sum(np.abs(ind))
    mse_list.append(mse_val)
    complexity_list.append(comp_val)


plt.figure(figsize=(6,5))
plt.scatter(complexity_list, mse_list, c='royalblue', edgecolor='k')
plt.xlabel("Complejidad")
plt.ylabel("Error de reconstrucción")
plt.title("Frente de Pareto - NSGA-II")
plt.grid(True)
plt.show()


best = best_individuals[0]
best_ae = SimpleAutoencoder(input_dim, latent_dim, np.array(best))
reconstructed = best_ae.forward(beats)

idx = np.random.randint(0, beats.shape[0])
plt.figure(figsize=(8,4))
plt.plot(beats[idx], label="Original", lw=2)
plt.plot(reconstructed[idx], '--', label="Reconstruida", lw=2)
plt.title(f"Reconstrucción del latido #{idx}")
plt.xlabel("Muestras")
plt.ylabel("Amplitud normalizada")
plt.legend()
plt.grid(True)
plt.show()


if logbook:
    gen = logbook.select("gen")
    min_mse = logbook.select("min")
    avg_mse = logbook.select("avg")

    plt.figure(figsize=(7,4))
    plt.plot(gen, min_mse, label="Mejor MSE", lw=2)
    plt.plot(gen, avg_mse, '--', label="Promedio MSE", lw=2)
    plt.xlabel("Generación")
    plt.ylabel("Error MSE")
    plt.title("Evolución del Error")
    plt.legend()
    plt.grid(True)
    plt.show()
