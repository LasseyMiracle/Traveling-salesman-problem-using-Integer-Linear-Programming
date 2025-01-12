# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 22:22:19 2024

@author: Miralce
"""

import pulp
import pandas as pd

def load_distance_matrix_from_excel(file_path, sheet_name):
    """Loads a distance matrix from an Excel file.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str): The name of the sheet containing the distance matrix. 

    Returns:
        numpy.ndarray: The distance matrix as a NumPy array.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, index_col=None)  # header=None to avoid using the first row as header
    distance_matrix = df.to_numpy()
    return distance_matrix

def solve_tsp(distance_matrix):
    """
    Solve the Traveling Salesman Problem using Integer Linear Programming (ILP).

    Args:
        distance_matrix (2D array): Symmetric matrix where element (i, j) is the distance between cities i and j.

    Returns:
        dict: Optimal tour and its total cost.
    """
    n = len(distance_matrix)
    
    # Define the problem
    tsp = pulp.LpProblem("Traveling_Salesman_Problem", pulp.LpMinimize)
    
    # Decision variables: x[i][j] = 1 if the path from i to j is used
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(n)] for i in range(n)]
    
    # Auxiliary variables for subtour elimination
    u = [pulp.LpVariable(f"u_{i}", lowBound=0, upBound=n-1, cat="Continuous") for i in range(n)]
    
    # Objective function: minimize the total distance
    tsp += pulp.lpSum(distance_matrix[i][j] * x[i][j] for i in range(n) for j in range(n) if i != j)
    
    # Constraints:
    # 1. Each city is entered exactly once
    for j in range(n):
        tsp += pulp.lpSum(x[i][j] for i in range(n) if i != j) == 1
    
    # 2. Each city is exited exactly once
    for i in range(n):
        tsp += pulp.lpSum(x[i][j] for j in range(n) if i != j) == 1
    
    # 3. Subtour elimination (MTZ constraints)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                tsp += u[i] - u[j] + (n - 1) * x[i][j] <= n - 2
    
    # Solve the problem
    tsp.solve()
    
    # Extract the solution
    tour = []
    if pulp.LpStatus[tsp.status] == "Optimal":
        total_cost = pulp.value(tsp.objective)
        for i in range(n):
            for j in range(n):
                if pulp.value(x[i][j]) == 1:
                    tour.append((i, j))
        return {"tour": tour, "total_cost": total_cost}
    else:
        return {"status": pulp.LpStatus[tsp.status]}

# Example usage
if __name__ == "__main__":
    file_path = r"C:/Users/Acer/Downloads/Distance_Matrix_TSP_Tours.xlsx"  # Replace with your file path
    sheet_name = input("Enter the sheet name: ")  # Get sheet name from user input
    distance_matrix = load_distance_matrix_from_excel(file_path, sheet_name)
    
    result = solve_tsp(distance_matrix)
    print("Optimal Tour:", result.get("tour"))
    print("Total Cost:", result.get("total_cost"))
