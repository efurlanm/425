# Stochastic vs Deterministic models

*Last edited: 2023-10-10*

Below are some points that I note as I read about the subject.

- Stochastic models and deterministic models are two distinct approaches used to describe and predict the behavior of systems, and the fundamental distinction lies in the level of uncertainty they consider.

- For a given set of inputs, a deterministic model will always produce the same result, while a stochastic model will produce varying results due to random fluctuations.

- The choice between a stochastic or deterministic model depends on the 
  nature of the system being studied, as well as the level of uncertainty 
  that is acceptable in the predictions made by the model.

| Stochastic Model                                                                                        | Deterministic Model                                                                                        |
| ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Incorporates randomness and uncertainty                                                                 | It incorporates determinism, certainty, predictability.                                                    |
| Produces different outputs for the same inputs                                                          | Produces the same output for the same inputs                                                               |
| Often uses probability distributions                                                                    | Often uses deterministic equations                                                                         |
| Useful for modeling systems with inherent randomness                                                    | Useful for modeling systems with known cause-and-effect relationships                                      |
| Appropriate for systems that inherently have randomness such as stock market, weather forecasting, etc. | More suitable for systems where cause-and-effect relationships are known, for example, engineering systems |
| Results can be difficult to interpret                                                                   | Simple to understand and implement                                                                         |
| The models can be mathematically complex                                                                | Can be easily solved mathematically                                                                        |

- PINNs encode model equations as a component of the NN.

- Generally used to solve PDE, but can solve SDE as well
  
     - PDE = Partial Differential Equation
     - SPDE = Stochastic Partial Differential Equation

- SPDE: one or more of the terms is a stochastic process.

- One of the most studied SPDEs is the Stochastic Heat Equation, which may formally be written as $\partial _{t}u=\Delta u+\xi$ , where $\Delta$ is the Laplacian and $\xi$ denotes space-time white noise.

## References

- <https://en.wikipedia.org/wiki/Stochastic_partial_differential_equation>
- <https://www.askpython.com/python/examples/deterministic-vs-stochastic-machine-learning>
- <https://en.wikipedia.org/wiki/Deterministic_system>
- <https://en.wikipedia.org/wiki/Stochastic_process>
- <https://mathvswild.com/what-is-a-deterministic-model/>
- <https://mathvswild.com/stochastic-vs-deterministic-models/>
