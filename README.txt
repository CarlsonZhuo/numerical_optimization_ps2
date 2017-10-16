Author: Jiacheng Zhuo
If you find it difficult to understand my codes, please let me know and I can demonstrate the codes for you

Two methods that I implemented:
(1) Gradient Descend
(2) BFGS

No special library required.

Special Notice:
(1) I negate the objective function so that I am still doing minimisation
(2) BFGS may fail due to the non-convexity of the objective function. I pass those cases when the approximated hessian matrix is singular, and return x immediately.

Special Notice for problem4:
(1) The gradient computation method is shown in person to the TA

EOF