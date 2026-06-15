"""StepCo (Wu et al., 2024) baseline — stepwise verify-then-revise with Math-Shepherd PRM.

Faithful re-implementation of the StepCo algorithm from
https://github.com/wzy6642/StepCo using Math-Shepherd
(peiyi9979/math-shepherd-mistral-7b-prm) as the process-supervised verifier
(this is also the default verifier in StepCo's released config).
"""
