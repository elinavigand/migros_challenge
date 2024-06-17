# Migros Store Challenge

## Project Objective
The purpose of this project is to find the best places to create new Migros stores.

### Team
* Daniel Karoly
* Elina Vigand
* Justin Villard
* Vahid Mamduhi

### Interactive Location Recommender
This tool helps you find the best location for a new Migros store in Zürich. Follow these steps:

1. **Understand the key performance indicators (KPIs):**
   - **Migros Market Share (a_1):** Number of Migros stores in a Kreis / Total number of supermarkets (the lower, the better).
   - **Current Population Density (a_2):** Current population density in the Kreis / Population density of Zürich (the higher, the better).
   - **Income Level (a_3):** Average income in the Kreis / Current average income in Zürich (the higher, the better).
   - **Future Population Density (a_4):** Prediction (2040) of population density change / Population density of Zürich (the higher, the better).

2. **Set the weights:**
   - Enter weights for each KPI according to their importance. Default weights are set to 1.00.
   - Adjust weights using the + or - buttons next to each KPI.

3. **Calculate the relevance factor:**
   - The tool uses a weighted linear model: \( \text{Relevance Factor} = a_1 \cdot w_1 + a_2 \cdot w_2 + a_3 \cdot w_3 + a_4 \cdot w_4 \)
   - The weights (w_i) are your input values, and the KPIs (a_i) are predefined.

4. **Get the recommendation:**
   - Based on the weights you set, the tool calculates the relevance factor for each Kreis.
   - The Kreis with the highest factor is the recommended location for the new Migros store.
####
**Example**

- If all weights are set to 1.0, the tool will equally consider all KPIs.
- In the example provided, the recommendation is to open a new Migros store in **Kreis 5**.
####
**App demo**

Feel free to test the [app demo](https://migroschallenge.streamlit.app/) and see how the recommendations change based on different priorities.

