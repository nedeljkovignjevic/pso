from src.ann_criterion import optimality_criterion, np
from src.pso import PSO


if __name__ == '__main__':

    # Example of evaluating ann
    w = np.random.uniform(-10, 10, 60)
    result = optimality_criterion(w)
    print(result)

    # Training ann (minimizing cost function)
    p = PSO(optimality_criterion, 60)
    p.optimize()
