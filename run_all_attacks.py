import attack_pro
import attack
import statistics

RUNS = 3

soft_results = []
hard_results = []

for _ in range(RUNS):
    soft_results.append(attack_pro.attack())

for _ in range(RUNS):
    hard_results.append(attack.attack())

print("Soft-label results:", soft_results)
print("Soft-label average:", statistics.mean(soft_results))
print("Soft-label std:", statistics.stdev(soft_results))

print("\nHard-label results:", hard_results)
print("Hard-label average:", statistics.mean(hard_results))
print("Hard-label std:", statistics.stdev(hard_results))