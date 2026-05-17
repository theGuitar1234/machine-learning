if __name__ == "__main__":

    import numpy as np

    X_full = [
        [4, 22],
        [6, 20],
        [8, 18],
        [10, 24],
        [12, 20],
        [24, 2],
        [20, 4],
        [26, 6],
        [30, 8],
        [28, 4],
        [40, 30],
        [44, 34],
        [42, 32],
        [38, 28],
        [46, 36],
    ]

    Y_full = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    X_test = [
        [5, 21],
        [7, 19],
        [9, 22],
        [11, 17],
        [13, 23],
        [22, 3],
        [25, 5],
        [27, 7],
        [29, 6],
        [31, 9],
        [39, 29],
        [41, 31],
        [43, 33],
        [45, 35],
        [47, 37],
    ]

    Y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    X_full = np.asarray(X_full, dtype=np.float32)
    Y_full = np.asarray(Y_full, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    Y_test = np.asarray(Y_test, dtype=np.float32)

    # np.savez(
    #     DATASET_PATH,
    #     X_full=X_full,
    #     Y_full=Y_full,
    #     X_test=X_test,
    #     Y_test=Y_test,
    # )
    # sys.exit(0)

    # DATASET_PATH = "trees/dataset.npz"

    # dataset = np.load(DATASET_PATH)

    # X_full = dataset["X_full"]
    # Y_full = dataset["Y_full"]
    # X_test = dataset["X_test"]
    # Y_test = dataset["Y_test"]

    # self = tree.fit(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    # )
    # _, train_accuracy = training_evaluation
    # _, test_accuracy = test_evaluation

    # print(f"\nTrain Accuracy : {train_accuracy}")
    # print(f"\nTest Accuracy : {test_accuracy}")

    # input = [
    #     [5.1, 3.5, 1.4, 0.2],
    #     [6.2, 2.8, 4.8, 1.8],
    # ]

    # print(f"\nSamples : {input}")
    # print(f"Predictions : {tree.predict(input)}")

    # test_predictions = tree.predict(X_test)

    # print("\nConfusion Matrix : ")
    # print(confusion_matrix(y_test, test_predictions))

    # X = [
    #     [2, 1],
    #     [2, 0],
    #     [2, 0],
    #     [1, 1],
    #     [1, 0],
    #     [0, 1],
    #     [0, 0],
    #     [0, 0],
    # ]
    # y = [1, 1, 1, 1, 0, 1, 0, 0]

    # X = [
    #     [60],
    #     [70],
    #     [75],
    #     [85],
    #     [90],
    #     [95],
    #     [100],
    #     [110],
    #     [120],
    #     [125],
    # ]
    # y = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]

    # best_val_accuracy, best_tree, best_config = ANonSeriousDecisionTree.validate_tree(
    #     configs
    # )

    # print("\nBest config:", best_config)
    # print("Best validation accuracy:", best_val_accuracy)

    # _, test_accuracy = best_tree.evaluate_dataset(X_test, y_test)
    # print(f"\nTest Accuracy: {test_accuracy}")




