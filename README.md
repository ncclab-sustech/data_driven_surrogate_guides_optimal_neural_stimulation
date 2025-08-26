# data-driven surrogate guides optimal neural stimulation

## **Requirements**

**Operating Systems**: Windows 10, Ubuntu 20.04

**Python Version**: 3.10+

**Non-standard Hardware Requirements**: No non-standard hardware is required to run this software.

**Note on PyTorch**: We recommend installing the GPU version of PyTorch, especially if you plan to simulate large datasets. 


## **How to run**

The notebook `.ipynb` files contain step-by-step instructions, including how to prepare your data, train a neural network as an IO surrogate brain, generate the reachable set by the grid search method, and surrogate guides optimal stimulation.

## **Demo examples**

Demos ``./toy_modesl/network_coupled_Wilson_Cowan_model/10fold_validation_in_wilson_cowan.ipynb` and `./toy_modesl/network_coupled_Wilson_Cowan_model/surrogate_guides_optimal_control_in_wilson_cowan.ipynb` show the application of predicting the generated neural data, generating and analyzing the reachable set on both the IO surrogate and the real dynamics, and guiding optimal stimulation with the IO surrogate. 


## **Contact**

For any questions/comments, please contact [NCC Lab](https://www.sustech.edu.cn/en/faculties/liuquanying.html).

## **Copyright**

Copyright © 2025 NCC Lab, Southern University of Science and Technology, Shenzhen, China.
