import json
from pathlib import Path
import matplotlib.pyplot as plt

log_path = Path("runs/tinystories_5k/checkpoint.trainlog.jsonl")  # change me

train_x, train_y = [], []
val_x, val_y = [], []

for line in log_path.read_text().splitlines():
    r = json.loads(line)
    if r["split"] == "train":
        train_x.append(r["iter"]); train_y.append(r["loss"])
    else:
        val_x.append(r["iter"]); val_y.append(r["loss"])

plt.figure()
plt.plot(train_x, train_y)
plt.title("Train loss")
plt.xlabel("iter")
plt.ylabel("loss")
plt.show()

plt.figure()
plt.plot(val_x, val_y)
plt.title("Val loss")
plt.xlabel("iter")
plt.ylabel("loss")
plt.show()
