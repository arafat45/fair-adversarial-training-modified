# Modified implementation of *"On the Alignment between Fairness and Accuracy: From the Perspective of Adversarial Robustness"*, extended with NES and Transfer black-box attacks alongside the paper's original PGD attack.

While numerous work has been proposed to address fairness in machine learning, existing methods do not guarantee fair predictions under imperceptible feature perturbation, and a seemingly fair model can suffer from large group-wise disparities under such perturbation. Moreover, while adversarial training has been shown to be reliable in improving a model’s robustness to defend against adversarial feature perturbation that deteriorates accuracy, it has not been properly studied in the context of adversarial perturbation against fairness. To tackle these challenges, in this paper, we study the problem of adversarial attack and adversarial robustness w.r.t. two terms: fairness and accuracy. From the adversarial attack perspective, we propose a unified structure for adversarial attacks against fairness which brings together common notions in group fairness, and we theoretically prove the equivalence of adversarial attacks against different fairness notions. Further, we derive the connections between adversarial attacks against fairness and those against accuracy. From the adversarial robustness perspective, we theoretically align robustness to adversarial attacks against fairness and accuracy, where robustness w.r.t. one term enhances robustness w.r.t. the other term. Our study suggests a novel way to unify adversarial training w.r.t. fairness and accuracy, and experiments show our proposed method achieves better robustness w.r.t. both terms.

# Step by Step Guide
01. Please install all necessary packages via: pip install -r requirements.txt
02. The required datasets: COMPAS and Adult are already in the repository. Keep them in the repository.
03. To get the results on COMPAS dataset. Run compas.py with arguments as follows:
   ```bash
python compas.py --csv compas-scores-two-years.csv
```
Optional flags (defaults shown):

```bash
python compas.py \
    --csv compas-scores-two-years.csv \
    --epochs 60 \
    --batch 256 \
    --seed 0 \
    --out compas_results.png \
    --out-attacks compas_attacks.png
```
04. To get the results on Adult dataset. Run adult.py with arguments as follows:

```bash
python adult.py --csv adult_reconstruction.csv
```

Same flag set, but the defaults are different (Adult is a larger dataset):

```bash
python adult.py \
    --csv adult_reconstruction.csv \
    --epochs 100 \
    --batch 128 \
    --seed 0 \
    --out adult_results.png \
    --out-attacks adult_attacks.png
```

05. Reading the outputs:

* `compas_results.png` / `adult_results.png` — 
  left = accuracy under PGD-acc, right = EOd under PGD-fair. 
* `compas_attacks.png` / `adult_attacks.png` —
* 3×2 grid: rows = attack
  (PGD / NES / Transfer), columns = (accuracy under acc-attack, EOd under
  fair-attack).
Note that in the output plot the adv+fair plot is the EOd / accuracy under fairness adversarial training.

# Software

| Package        | Version  | 
| -------------- | -------- | 
| Python         | 3.10.12  | 
| PyTorch        | 2.2.1    | 
| torchvision    | 0.17.1   | 
| NumPy          | 1.26.4   |
| pandas         | 2.2.1    | 
| scikit-learn   | 1.4.1    | 
| matplotlib     | 3.8.3    |
| CUDA Toolkit   | 11.8     |

# Citation 

Code repository is forked from the original implementation of the paper "On the Alignment between Fairness and Accuracy: from the Perspective of Adversarial Robustness", published in ICML 2026. Please refer the original code repository: https://github.com/cjy24/fair-adversarial-training

# Contact

For questions or concerns please contact at: abdullaharafat.miah@uri.edu

