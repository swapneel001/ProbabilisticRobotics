import numpy as np
import matplotlib.pyplot as plt
#measurement model
M = np.array([
    [0.9, 0.5],  
    [0.1, 0.5]   
])

#action model
M_Push = np.array([
    [1.0, 0.6], 
    [0.0, 0.4]  
])

#initial conditions
bel = np.array([[0.5], [0.5]])

measurements = [0, 0, 0, 0, 0, 1, 0, 0]

history_open = [bel[0,0]]
history_closed = [bel[1,0]]

#running filter with action
for t, measurement in enumerate(measurements):
    bel_bar = M_Push @ bel #apply action model
    # measurement update
    likelihood = M[measurement].reshape(2,1)
    unnormalized_posterior = likelihood * bel_bar
    bel = unnormalized_posterior / unnormalized_posterior.sum()
    history_open.append(bel[0,0])
    history_closed.append(bel[1,0])
    print(f"step {t}: bel(open)={bel[0,0]:.3f}, bel(closed)={bel[1,0]:.3f}")

bel = [0.5, 0.5]  # reset belief
history_open_noaction = [bel[0]]
history_closed_noaction = [bel[1]]

for t,m in enumerate(measurements):
    # measurement update only
    likelihood = M[m]
    unnormalized_posterior = likelihood * bel
    bel = unnormalized_posterior / np.sum(unnormalized_posterior)
    history_open_noaction.append(bel[0])
    history_closed_noaction.append(bel[1])
    print(f"step {t} (no action): bel(open)={bel[0]:.3f}, bel(closed)={bel[1]:.3f}")

steps = range(len(history_open))
plt.figure(figsize=(8,5))
plt.plot(steps, history_open, marker='o', label="bel(open) with push action")
plt.plot(steps,history_open_noaction, marker='x', label="bel(open) no action")
plt.xlabel("Time step")
plt.ylabel("Belief")
plt.title("Belief progression with and without action")
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.show()