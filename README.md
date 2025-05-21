# On the Alignment between Fairness and Accuracy: from the Perspective of Adversarial Robustness

While numerous work has been proposed to ad-
dress fairness in machine learning, existing meth-
ods do not guarantee fair predictions under im-
perceptible feature perturbation, and a seemingly
fair model can suffer from large group-wise dis-
parities under such perturbation. Moreover, while
adversarial training has been shown to be reli-
able in improving a model’s robustness to defend
against adversarial feature perturbation that dete-
riorates accuracy, it has not been properly studied
in the context of adversarial perturbation against
fairness. To tackle these challenges, in this paper,
we study the problem of adversarial attack and ad-
versarial robustness w.r.t. two terms: fairness and
accuracy. From the adversarial attack perspec-
tive, we propose a unified structure for adversar-
ial attacks against fairness which brings together
common notions in group fairness, and we theoret-
ically prove the equivalence of adversarial attacks
against different fairness notions. Further, we de-
rive the connections between adversarial attacks
against fairness and those against accuracy. From
the adversarial robustness perspective, we theo-
retically align robustness to adversarial attacks
against fairness and accuracy, where robustness
w.r.t. one term enhances robustness w.r.t. the
other term. Our study suggests a novel way to
unify adversarial training w.r.t. fairness and accu-
racy, and experiments show our proposed method
achieves better robustness w.r.t. both terms.

# Configuration
 Please install all necessary packages via: pip install -r requirements.txt

 #  Usage
 The example contains experiments on COMPAS   dataset. To run the experiments, please revise the file directory accordingly.
