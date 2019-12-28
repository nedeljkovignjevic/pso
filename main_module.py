from src.ann_criterion import optimality_criterion, np
from src.pso import PSO


if __name__ == '__main__':

    # Evaluate ANN performance for the given set of random weights
    w = np.random.uniform(-10, 10, 60)
    result = optimality_criterion(w)
    print(f'Random weights: {w}\nEvaluation of ANN performance: {result}\n')

    # Train ANN (minimize loss function)
    p = PSO(optimality_criterion, 60)
    w = p.optimize()
    result = optimality_criterion(w)
    print(f'\nOptimized weights: {w}\nEvaluation of ANN performance: {result}')