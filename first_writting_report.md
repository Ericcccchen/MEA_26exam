/Users/x/anaconda3/envs/python_pytorch2/bin/python /Users/x/pycharm_project/python_text2_pytorch_3/train5_3.py
Using device: mps
Training set: 13,049, Validation set: 11,488
Number of parameters: 21,417,542

======================================================================
 Improved Scheme v3.3 — ResNet34 + FocalLoss + 160x160
======================================================================
  Input: 160x160
  Class weights: ['1.770', '1.438', '0.929', '0.789', '0.704', '1.026']
  Warmup: 5 epochs (without Mixup)
  Main training: 75 epochs (Mixup α=0.4)
  Backbone: ResNet34 (21M parameters)
  Loss: FocalLoss γ=2.0 + class weights
  Checkpoint: checkpoint_v3_focal.pth
======================================================================

Training from scratch
[W] Epoch [  1/80] 129.7s  LR: 1.0e-05/5.0e-04
  Train Loss: 0.7765, Acc: 41.63%
  Val   Loss: 0.7873, Acc: 53.47%  ✓ New best: 53.47%
  Per class: anger:46.1%  disgust:78.8%  fear:85.7%  happy:57.0%  neutral:30.2%  sad:5.8%
----------------------------------------------------------------------
[W] Epoch [  2/80] 125.7s  LR: 1.0e-05/5.0e-04
  Train Loss: 0.5454, Acc: 55.02%
  Val   Loss: 0.8838, Acc: 60.75%  ✓ New best: 60.75%
  Per class: anger:62.5%  disgust:53.7%  fear:89.9%  happy:72.6%  neutral:33.1%  sad:32.9%
----------------------------------------------------------------------
[W] Epoch [  3/80] 123.3s  LR: 1.0e-05/5.0e-04
  Train Loss: 0.4754, Acc: 61.74%
  Val   Loss: 0.8812, Acc: 65.87%  ✓ New best: 65.87%
  Per class: anger:60.1%  disgust:38.1%  fear:90.8%  happy:82.1%  neutral:64.8%  sad:39.6%
----------------------------------------------------------------------
[W] Epoch [  4/80] 123.7s  LR: 9.9e-06/5.0e-04
  Train Loss: 0.4299, Acc: 66.37%
  Val   Loss: 0.8833, Acc: 66.77%  ✓ New best: 66.77%
  Per class: anger:61.3%  disgust:69.5%  fear:93.1%  happy:81.9%  neutral:47.8%  sad:27.9%
----------------------------------------------------------------------
[W] Epoch [  5/80] 122.2s  LR: 9.9e-06/5.0e-04
  Train Loss: 0.3975, Acc: 68.96%
  Val   Loss: 0.9079, Acc: 69.82%  ✓ New best: 69.82%
  Per class: anger:63.8%  disgust:58.7%  fear:91.3%  happy:87.5%  neutral:59.1%  sad:40.2%
----------------------------------------------------------------------

[Main training] backbone LR → 5e-05, Mixup enabled
    Epoch [  6/80] 124.1s  LR: 5.0e-05/4.9e-04
  Train Loss: 0.5850, Acc: 56.59%
  Val   Loss: 0.6049, Acc: 66.54%  (Best: 69.82%, 1/20)
  Per class: anger:44.8%  disgust:35.8%  fear:96.9%  happy:76.0%  neutral:58.3%  sad:59.7%
----------------------------------------------------------------------
    Epoch [  7/80] 122.9s  LR: 5.0e-05/4.9e-04
  Train Loss: 0.5254, Acc: 57.82%
  Val   Loss: 0.5900, Acc: 72.98%  ✓ New best: 72.98%
  Per class: anger:65.7%  disgust:45.0%  fear:87.8%  happy:97.3%  neutral:81.4%  sad:44.1%
----------------------------------------------------------------------
    Epoch [  8/80] 122.6s  LR: 4.9e-05/4.9e-04
  Train Loss: 0.5107, Acc: 59.06%
  Val   Loss: 0.4533, Acc: 72.02%  (Best: 72.98%, 1/20)
  Per class: anger:60.0%  disgust:55.2%  fear:97.6%  happy:76.4%  neutral:78.0%  sad:48.1%
----------------------------------------------------------------------
    Epoch [  9/80] 128.3s  LR: 4.9e-05/4.8e-04
  Train Loss: 0.4642, Acc: 61.18%
  Val   Loss: 0.4523, Acc: 73.87%  ✓ New best: 73.87%
  Per class: anger:63.2%  disgust:63.7%  fear:95.7%  happy:86.5%  neutral:75.8%  sad:41.6%
----------------------------------------------------------------------
    Epoch [ 10/80] 124.9s  LR: 4.9e-05/4.8e-04
  Train Loss: 0.4472, Acc: 64.55%
  Val   Loss: 0.4627, Acc: 74.95%  ✓ New best: 74.95%
  Per class: anger:76.7%  disgust:49.2%  fear:92.9%  happy:93.0%  neutral:74.6%  sad:48.9%
----------------------------------------------------------------------
    Epoch [ 11/80] 123.6s  LR: 4.8e-05/4.8e-04
  Train Loss: 0.4335, Acc: 63.26%
  Val   Loss: 0.4907, Acc: 75.15%  ✓ New best: 75.15%
  ✓ Surpassed 75%!
  Per class: anger:73.3%  disgust:59.7%  fear:90.6%  happy:96.3%  neutral:74.9%  sad:42.1%
----------------------------------------------------------------------
    Epoch [ 12/80] 123.4s  LR: 4.8e-05/4.7e-04
  Train Loss: 0.4214, Acc: 65.36%
  Val   Loss: 0.4922, Acc: 75.50%  ✓ New best: 75.50%
  ✓ Surpassed 75%!
  Per class: anger:67.1%  disgust:49.3%  fear:92.2%  happy:95.3%  neutral:89.3%  sad:44.2%
----------------------------------------------------------------------
    Epoch [ 13/80] 124.5s  LR: 4.7e-05/4.7e-04
  Train Loss: 0.4057, Acc: 63.52%
  Val   Loss: 0.4981, Acc: 75.99%  ✓ New best: 75.99%
  ✓ Surpassed 75%!
  Per class: anger:67.5%  disgust:60.2%  fear:92.9%  happy:94.3%  neutral:80.7%  sad:44.9%
----------------------------------------------------------------------
    Epoch [ 14/80] 124.3s  LR: 4.7e-05/4.6e-04
  Train Loss: 0.3928, Acc: 64.52%
  Val   Loss: 0.4904, Acc: 75.30%  (Best: 75.99%, 1/20)
  Per class: anger:65.1%  disgust:50.6%  fear:96.4%  happy:87.6%  neutral:87.8%  sad:48.1%
----------------------------------------------------------------------
    Epoch [ 15/80] 122.8s  LR: 4.6e-05/4.6e-04
  Train Loss: 0.3746, Acc: 66.65%
  Val   Loss: 0.5180, Acc: 76.19%  ✓ New best: 76.19%
  ✓ Surpassed 75%!
  Per class: anger:72.0%  disgust:45.0%  fear:93.3%  happy:94.6%  neutral:86.6%  sad:50.6%
----------------------------------------------------------------------
    Epoch [ 16/80] 123.0s  LR: 4.6e-05/4.5e-04
  Train Loss: 0.3465, Acc: 66.01%
  Val   Loss: 0.4747, Acc: 75.92%  (Best: 76.19%, 1/20)
  Per class: anger:67.9%  disgust:53.8%  fear:91.5%  happy:95.2%  neutral:83.9%  sad:48.1%
----------------------------------------------------------------------
    Epoch [ 17/80] 122.2s  LR: 4.5e-05/4.5e-04
  Train Loss: 0.3448, Acc: 69.02%
  Val   Loss: 0.4980, Acc: 76.04%  (Best: 76.19%, 2/20)
  Per class: anger:68.5%  disgust:52.3%  fear:90.3%  happy:96.6%  neutral:80.8%  sad:52.1%
----------------------------------------------------------------------
    Epoch [ 18/80] 122.3s  LR: 4.4e-05/4.4e-04
  Train Loss: 0.3471, Acc: 65.28%
  Val   Loss: 0.4871, Acc: 75.03%  (Best: 76.19%, 3/20)
  Per class: anger:59.6%  disgust:52.9%  fear:96.9%  happy:86.6%  neutral:90.0%  sad:46.8%
----------------------------------------------------------------------
    Epoch [ 19/80] 123.0s  LR: 4.4e-05/4.3e-04
  Train Loss: 0.2949, Acc: 67.97%
  Val   Loss: 0.5773, Acc: 75.76%  (Best: 76.19%, 4/20)
  Per class: anger:66.2%  disgust:36.4%  fear:93.2%  happy:94.6%  neutral:92.6%  sad:54.3%
----------------------------------------------------------------------
    Epoch [ 20/80] 122.9s  LR: 4.3e-05/4.3e-04
  Train Loss: 0.2989, Acc: 71.29%
  Val   Loss: 0.5302, Acc: 76.37%  ✓ New best: 76.37%
  ✓ Surpassed 75%!
  Per class: anger:63.1%  disgust:56.9%  fear:92.8%  happy:94.3%  neutral:91.8%  sad:44.0%
----------------------------------------------------------------------
    Epoch [ 21/80] 319.0s  LR: 4.2e-05/4.2e-04
  Train Loss: 0.2979, Acc: 69.54%
  Val   Loss: 0.5459, Acc: 76.11%  (Best: 76.37%, 1/20)
  Per class: anger:63.9%  disgust:56.6%  fear:94.2%  happy:93.8%  neutral:83.4%  sad:47.7%
----------------------------------------------------------------------
    Epoch [ 22/80] 2233.4s  LR: 4.2e-05/4.1e-04
  Train Loss: 0.2646, Acc: 70.48%
  Val   Loss: 0.6338, Acc: 76.30%  (Best: 76.37%, 2/20)
  Per class: anger:59.5%  disgust:54.3%  fear:96.4%  happy:89.2%  neutral:91.9%  sad:49.1%
----------------------------------------------------------------------
    Epoch [ 23/80] 137.5s  LR: 4.1e-05/4.0e-04
  Train Loss: 0.2698, Acc: 70.20%
  Val   Loss: 0.6049, Acc: 76.43%  ✓ New best: 76.43%
  ✓ Surpassed 75%!
  Per class: anger:74.7%  disgust:40.5%  fear:91.7%  happy:96.1%  neutral:89.3%  sad:52.3%
----------------------------------------------------------------------
    Epoch [ 24/80] 129.4s  LR: 4.0e-05/4.0e-04
  Train Loss: 0.2790, Acc: 72.98%
  Val   Loss: 0.5254, Acc: 76.36%  (Best: 76.43%, 1/20)
  Per class: anger:64.8%  disgust:61.7%  fear:94.6%  happy:92.9%  neutral:94.7%  sad:35.4%
----------------------------------------------------------------------
    Epoch [ 25/80] 124.6s  LR: 3.9e-05/3.9e-04
  Train Loss: 0.2598, Acc: 72.83%
  Val   Loss: 0.5850, Acc: 76.38%  (Best: 76.43%, 2/20)
  Per class: anger:67.8%  disgust:45.9%  fear:92.9%  happy:94.5%  neutral:94.3%  sad:48.0%
----------------------------------------------------------------------
    Epoch [ 26/80] 124.7s  LR: 3.8e-05/3.8e-04
  Train Loss: 0.2713, Acc: 73.58%
  Val   Loss: 0.5371, Acc: 76.71%  ✓ New best: 76.71%
  ✓ Surpassed 75%!
  Per class: anger:62.5%  disgust:54.1%  fear:94.0%  happy:93.5%  neutral:96.0%  sad:44.5%
----------------------------------------------------------------------
    Epoch [ 27/80] 129.8s  LR: 3.8e-05/3.7e-04
  Train Loss: 0.2364, Acc: 74.62%
  Val   Loss: 0.5916, Acc: 76.15%  (Best: 76.71%, 1/20)
  Per class: anger:58.7%  disgust:50.4%  fear:92.1%  happy:95.5%  neutral:97.2%  sad:46.1%
----------------------------------------------------------------------
    Epoch [ 28/80] 125.2s  LR: 3.7e-05/3.6e-04
  Train Loss: 0.2554, Acc: 71.14%
  Val   Loss: 0.5741, Acc: 77.61%  ✓ New best: 77.61%
  ✓ Surpassed 75%!
  Per class: anger:66.7%  disgust:62.3%  fear:92.1%  happy:95.7%  neutral:93.9%  sad:41.8%
----------------------------------------------------------------------
    Epoch [ 29/80] 124.5s  LR: 3.6e-05/3.5e-04
  Train Loss: 0.2522, Acc: 75.25%
  Val   Loss: 0.5442, Acc: 76.55%  (Best: 77.61%, 1/20)
  Per class: anger:63.3%  disgust:45.6%  fear:92.7%  happy:95.0%  neutral:96.1%  sad:50.3%
----------------------------------------------------------------------
    Epoch [ 30/80] 577.3s  LR: 3.5e-05/3.5e-04
  Train Loss: 0.2535, Acc: 73.61%
  Val   Loss: 0.4744, Acc: 77.40%  (Best: 77.61%, 2/20)
  Per class: anger:62.7%  disgust:58.1%  fear:92.7%  happy:95.3%  neutral:94.0%  sad:46.4%
----------------------------------------------------------------------
    Epoch [ 31/80] 126.2s  LR: 3.4e-05/3.4e-04
  Train Loss: 0.2483, Acc: 73.55%
  Val   Loss: 0.7012, Acc: 76.16%  (Best: 77.61%, 3/20)
  Per class: anger:59.6%  disgust:44.8%  fear:94.9%  happy:92.1%  neutral:96.4%  sad:51.2%
----------------------------------------------------------------------
    Epoch [ 32/80] 126.2s  LR: 3.3e-05/3.3e-04
  Train Loss: 0.2381, Acc: 72.75%
  Val   Loss: 0.6575, Acc: 76.42%  (Best: 77.61%, 4/20)
  Per class: anger:58.7%  disgust:52.3%  fear:92.7%  happy:95.5%  neutral:98.2%  sad:44.3%
----------------------------------------------------------------------
    Epoch [ 33/80] 130.1s  LR: 3.2e-05/3.2e-04
  Train Loss: 0.2523, Acc: 74.12%
  Val   Loss: 0.7103, Acc: 76.26%  (Best: 77.61%, 5/20)
  Per class: anger:59.1%  disgust:54.4%  fear:90.2%  happy:97.2%  neutral:97.9%  sad:43.1%
----------------------------------------------------------------------
    Epoch [ 34/80] 128.2s  LR: 3.1e-05/3.1e-04
  Train Loss: 0.2215, Acc: 74.06%
  Val   Loss: 0.5260, Acc: 76.53%  (Best: 77.61%, 6/20)
  Per class: anger:58.9%  disgust:57.5%  fear:93.2%  happy:94.3%  neutral:96.6%  sad:42.6%
----------------------------------------------------------------------
    Epoch [ 35/80] 126.4s  LR: 3.0e-05/3.0e-04
  Train Loss: 0.2341, Acc: 74.86%
  Val   Loss: 0.5995, Acc: 77.67%  ✓ New best: 77.67%
  ✓ Surpassed 75%!
  Per class: anger:59.9%  disgust:63.8%  fear:94.8%  happy:92.6%  neutral:95.7%  sad:43.8%
----------------------------------------------------------------------
    Epoch [ 36/80] 124.8s  LR: 2.9e-05/2.9e-04
  Train Loss: 0.2061, Acc: 73.85%
  Val   Loss: 0.5787, Acc: 77.42%  (Best: 77.67%, 1/20)
  Per class: anger:67.5%  disgust:51.2%  fear:95.0%  happy:92.6%  neutral:94.1%  sad:49.3%
----------------------------------------------------------------------
    Epoch [ 37/80] 124.0s  LR: 2.8e-05/2.8e-04
  Train Loss: 0.2124, Acc: 75.37%
  Val   Loss: 0.6672, Acc: 76.64%  (Best: 77.67%, 2/20)
  Per class: anger:62.8%  disgust:47.5%  fear:95.8%  happy:91.2%  neutral:98.4%  sad:48.0%
----------------------------------------------------------------------
    Epoch [ 38/80] 124.2s  LR: 2.7e-05/2.7e-04
  Train Loss: 0.2172, Acc: 77.54%
  Val   Loss: 0.5376, Acc: 77.41%  (Best: 77.67%, 3/20)
  Per class: anger:62.3%  disgust:53.8%  fear:94.6%  happy:92.3%  neutral:95.3%  sad:50.3%
----------------------------------------------------------------------
    Epoch [ 39/80] 124.6s  LR: 2.6e-05/2.6e-04
  Train Loss: 0.2259, Acc: 74.91%
  Val   Loss: 0.6829, Acc: 76.71%  (Best: 77.67%, 4/20)
  Per class: anger:60.1%  disgust:46.9%  fear:95.0%  happy:92.4%  neutral:98.2%  sad:50.5%
----------------------------------------------------------------------
    Epoch [ 40/80] 128.1s  LR: 2.5e-05/2.5e-04
  Train Loss: 0.1791, Acc: 80.75%
  Val   Loss: 0.7928, Acc: 77.07%  (Best: 77.67%, 5/20)
  Per class: anger:62.7%  disgust:57.0%  fear:91.5%  happy:96.4%  neutral:98.1%  sad:42.4%
----------------------------------------------------------------------
    Epoch [ 41/80] 130.2s  LR: 2.4e-05/2.4e-04
  Train Loss: 0.2032, Acc: 75.43%
  Val   Loss: 0.6876, Acc: 76.58%  (Best: 77.67%, 6/20)
  Per class: anger:57.0%  disgust:51.6%  fear:94.7%  happy:92.7%  neutral:98.2%  sad:47.8%
----------------------------------------------------------------------
    Epoch [ 42/80] 127.1s  LR: 2.3e-05/2.3e-04
  Train Loss: 0.1942, Acc: 79.49%
  Val   Loss: 0.6895, Acc: 76.85%  (Best: 77.67%, 7/20)
  Per class: anger:57.2%  disgust:52.3%  fear:94.1%  happy:93.5%  neutral:98.9%  sad:47.8%
----------------------------------------------------------------------
    Epoch [ 43/80] 177.6s  LR: 2.2e-05/2.2e-04
  Train Loss: 0.2103, Acc: 76.87%
  Val   Loss: 0.7056, Acc: 76.78%  (Best: 77.67%, 8/20)
  Per class: anger:55.5%  disgust:49.7%  fear:94.0%  happy:94.3%  neutral:99.3%  sad:49.7%
----------------------------------------------------------------------
    Epoch [ 44/80] 356.8s  LR: 2.1e-05/2.1e-04
  Train Loss: 0.1939, Acc: 78.42%
  Val   Loss: 0.6808, Acc: 76.60%  (Best: 77.67%, 9/20)
  Per class: anger:63.9%  disgust:43.0%  fear:91.5%  happy:96.5%  neutral:99.2%  sad:49.9%
----------------------------------------------------------------------
    Epoch [ 45/80] 374.4s  LR: 2.0e-05/2.0e-04
  Train Loss: 0.2110, Acc: 75.85%
  Val   Loss: 0.7998, Acc: 76.27%  (Best: 77.67%, 10/20)
  Per class: anger:59.1%  disgust:49.2%  fear:93.1%  happy:95.4%  neutral:99.1%  sad:44.8%
----------------------------------------------------------------------
    Epoch [ 46/80] 196.8s  LR: 1.9e-05/1.9e-04
  Train Loss: 0.1931, Acc: 74.83%
  Val   Loss: 0.6303, Acc: 76.78%  (Best: 77.67%, 11/20)
  Per class: anger:60.4%  disgust:53.0%  fear:90.9%  happy:96.9%  neutral:99.1%  sad:44.8%
----------------------------------------------------------------------
    Epoch [ 47/80] 120.7s  LR: 1.8e-05/1.8e-04
  Train Loss: 0.2004, Acc: 77.51%
  Val   Loss: 0.5583, Acc: 77.26%  (Best: 77.67%, 12/20)
  Per class: anger:59.7%  disgust:51.3%  fear:94.7%  happy:93.3%  neutral:98.9%  sad:49.0%
----------------------------------------------------------------------
    Epoch [ 48/80] 119.1s  LR: 1.8e-05/1.7e-04
  Train Loss: 0.2077, Acc: 78.23%
  Val   Loss: 0.6506, Acc: 76.73%  (Best: 77.67%, 13/20)
  Per class: anger:59.5%  disgust:49.2%  fear:93.7%  happy:94.4%  neutral:99.2%  sad:47.6%
----------------------------------------------------------------------


---

/Users/x/anaconda3/envs/python_pytorch2/bin/python /Users/x/pycharm_project/python_text2_pytorch_3/train5_3.py
Using device: mps
Training set: 13,049, Validation set: 11,488
Number of parameters: 21,417,542

======================================================================
 Improved Scheme v3.3 — ResNet34 + FocalLoss + 160x160
======================================================================
  Input: 160x160
  Class weights: ['1.770', '1.438', '0.929', '0.789', '0.704', '1.026']
  Warmup: 5 epochs (without Mixup)
  Main training: 75 epochs (Mixup α=0.4)
  Backbone: ResNet34 (21M parameters)
  Loss: FocalLoss γ=2.0 + class weights
  Checkpoint: checkpoint_v3_focal.pth
======================================================================

✓ Resumed: epoch 48, best 77.67%
    Epoch [ 49/80] 139.2s  LR: 1.7e-05/1.6e-04
  Train Loss: 0.1914, Acc: 78.73%
  Val   Loss: 0.6082, Acc: 76.38%  (Best: 77.67%, 14/20)
  Per class: anger:57.5%  disgust:46.7%  fear:94.7%  happy:93.1%  neutral:99.1%  sad:49.2%
----------------------------------------------------------------------
    Epoch [ 50/80] 139.4s  LR: 1.6e-05/1.5e-04
  Train Loss: 0.1875, Acc: 78.20%
  Val   Loss: 0.6794, Acc: 76.53%  (Best: 77.67%, 15/20)
  Per class: anger:61.4%  disgust:48.6%  fear:91.9%  happy:96.1%  neutral:99.2%  sad:46.0%
----------------------------------------------------------------------
    Epoch [ 51/80] 139.2s  LR: 1.5e-05/1.5e-04
  Train Loss: 0.1922, Acc: 76.09%
  Val   Loss: 0.6674, Acc: 76.49%  (Best: 77.67%, 16/20)
  Per class: anger:60.9%  disgust:44.2%  fear:92.5%  happy:95.4%  neutral:99.6%  sad:49.8%
----------------------------------------------------------------------
    Epoch [ 52/80] 138.3s  LR: 1.4e-05/1.4e-04
  Train Loss: 0.1847, Acc: 76.96%
  Val   Loss: 0.6664, Acc: 76.73%  (Best: 77.67%, 17/20)
  Per class: anger:64.9%  disgust:44.6%  fear:91.5%  happy:96.4%  neutral:99.2%  sad:48.8%
----------------------------------------------------------------------
    Epoch [ 53/80] 137.8s  LR: 1.3e-05/1.3e-04
  Train Loss: 0.1977, Acc: 77.17%
  Val   Loss: 0.6299, Acc: 76.60%  (Best: 77.67%, 18/20)
  Per class: anger:61.9%  disgust:42.2%  fear:93.9%  happy:93.8%  neutral:99.2%  sad:51.8%
----------------------------------------------------------------------
    Epoch [ 54/80] 137.3s  LR: 1.2e-05/1.2e-04
  Train Loss: 0.1825, Acc: 80.08%
  Val   Loss: 0.7242, Acc: 76.90%  (Best: 77.67%, 19/20)
  Per class: anger:62.2%  disgust:50.9%  fear:95.1%  happy:92.2%  neutral:99.5%  sad:45.9%
----------------------------------------------------------------------
    Epoch [ 55/80] 136.7s  LR: 1.1e-05/1.1e-04
  Train Loss: 0.1950, Acc: 77.87%
  Val   Loss: 0.7154, Acc: 76.67%  (Best: 77.67%, 20/20)
  Per class: anger:62.2%  disgust:44.0%  fear:92.3%  happy:95.5%  neutral:99.6%  sad:50.3%
----------------------------------------------------------------------

Early stopping triggered! No improvement for 20 epochs

Training complete! Total time: 16.2 minutes
Best validation accuracy: 77.67%

Process finished with exit code 0

======================================================================
 Improved Scheme v3.3 — ResNet34 + FocalLoss + 160x160
======================================================================
  Input: 160x160
  Class weights: ['1.770', '1.438', '0.929', '0.789', '0.704', '1.026']
  Warmup: 5 epochs (without Mixup)
  Main training: 75 epochs (Mixup α=0.4)
  Backbone: ResNet34 (21M parameters)
  Loss: FocalLoss γ=2.0 + class weights
  Checkpoint: checkpoint_v3_focal.pth
======================================================================

✓ Resumed: epoch 55, best 77.67%
    Epoch [ 56/80] 138.9s  LR: 1.0e-05/1.0e-04
  Train Loss: 0.1900, Acc: 77.78%
  Val   Loss: 0.6985, Acc: 76.64%  (Best: 77.67%, 21/20)
  Per class: anger:61.4%  disgust:47.1%  fear:94.0%  happy:93.9%  neutral:99.7%  sad:47.4%
----------------------------------------------------------------------

Early stopping triggered! No improvement for 20 epochs

Training complete! Total time: 2.3 minutes
Best validation accuracy: 77.67%

Process finished with exit code 0