# Knowledge-Distillation-Keras
A pretty sloppy implementation of the [original paper](https://arxiv.org/pdf/1503.02531.pdf).


## Before you start!
This implementation is a test just after reading some papers, is not verified or corrected (yet) so don't take anything for granted. If you find a mistake in the code or in the understanding, please leave a comment or start an issue.

## Explanation
1. First you train a big model. In my case [VGG19](https://arxiv.org/abs/1409.1556) with the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). This is the teacher. The model has **38,947,914 parameters**. Damn.
2. It is very interesting/important to apply EarlyStopping to the teacher model. After about 20 epochs, the model stopped with a validation accuracy of around **66%**.
3. Create a smaller model, the student, (the architecture doesn't really have to match the teacher one) and train it the same way but applying the [KD-Loss](https://images2.programmersought.com/264/34/34c516ba0956a240c9641ccda709e160.png). This takes in account the softed logits of the teacher and student, and the current logits of the student.
4. After 17 epochs, the student model reached an accuracy of **60%** but using **103,050 parameters**. That is 9.1% less accuracy but 99.7% less parameters.

FYI: The student model alone had an accuracy of around 40%.
