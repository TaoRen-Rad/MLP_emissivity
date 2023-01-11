# MLP_emissivity

## Introduction
Accurate prediction of the total gray gas emissivity is important for many engineering applications where the radiative heat transfer between gases and wall surfaces needs to be estimated quickly. In this work, we propose a machine-learning approach for predicting the total gray gas emissivity of mixtures of H2O, CO2, CO, and N2 at a range of pressures. The model was trained using a dataset generated from the HITEMP-2010 spectroscopic database, and the Alberti-cut-off line-shape model was used to calculate the absorption coefficients and total gray gas emissivity. Our model demonstrated good accuracy, with a maximum relative error of less than 2.2 % and a relative error of no more than 0.603 % for 99.73 % of the thermodynamic states within the input range. Additionally, our model is compact, easy to use, and outperforms traditional methods that rely on interpolation from two-dimensional look-up tables and empirical correlations for mixtures。

The MLP-based model is developed for predicting the total gray gas emissivity of H2O-CO2-CO-N2 mixtures. The model's accuracy was evaluated across a range of pressures (0.1-80.0~bar), temperatures (300.0-3000.0~K), optical path lengths (0.1-6000.0~bar cm), and volume fractions (0.0-1.0) and found to have a maximum relative error of less than 2.2 % and a relative error of 99.73 % of thermodynamic states within the input range to be less than 0.607 %. Compared to the traditional emissivity chart method, the new model is both more efficient and accurate.

## Dependencies

Python: `numpy`, `mamtplotlib`, `labellines`, `tensorflow`

## Quick Start

Check the `example.ipynb`.
