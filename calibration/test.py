if __name__ == "__main__":
    # dataset = np.load("neural_network/data/npz/Binary_Train.npz")
    # X = dataset["X"]
    # y = dataset["Y"]

    # X, y = create_categorical_dataset(n_samples=300, seed=seed)
    
    # X = np.asarray(X)
    # Y = np.asarray(y)
    
    # X_train_val, X_test, Y_train_val, Y_test = train_test_split(
    #     X, 
    #     Y, 
    #     test_size=0.20, 
    #     random_state=seed, 
    #     stratify=Y.ravel()
    # )

    # X_train, X_valid, Y_train, Y_valid = train_test_split(
    #     X_train_val,
    #     Y_train_val,
    #     test_size=0.25,
    #     random_state=seed,
    #     stratify=Y_train_val.ravel(),
    # )
    
    # loaded_model = CatBoost(
    #     boosting_rounds=50,
    #     max_depth=15,
    #     information_gain=ANonSeriousDecisionTree.InformationGain.GINI,
    #     loss_type=CatBoost.LossType.SSE,
    #     restore_best=True,
    #     validation=True,
    #     early_stopping=True,
    #     sub_sample=1,
    #     column_sub_sample=1,
    #     symmetrical=False,
    #     boosting_type=CatBoost.BoostingType.PLAIN,
    # )
    # loaded_model.fit(X_train, Y_train, X_valid, Y_valid)
    
    model_name = "trained_model_digit_recognizer"
    loaded_model, _ = NeuralNetwork.load_model(
        f"neural_network/models/{model_name}.pkl", device=NeuralNetwork.Device.CPU
    )

    prepared_dataset = NeuralNetwork.load_from_npz("neural_network/data/npz/MNIST.npz")
    X_valid = prepared_dataset["X_valid"]
    Y_valid = prepared_dataset["Y_valid"]
    
    X_valid = X_valid.astype(np.float32) / 255.0
    
    # def spiral_of_clouds(
    #     n_objects_by_class,
    #     radius=5,
    #     n_turns=3,
    #     sigma=0.12,
    #     seed=0,
    #     angle=0,
    #     b=0.18,
    # ):
    #     rng = np.random.default_rng(seed)

    #     theta = np.linspace(0, 2 * np.pi * n_turns, n_objects_by_class)

    #     r = np.exp(b * theta) - 1
    #     r = radius * r / r.max()

    #     def arm(k):
    #         t = theta + k * np.pi + angle

    #         x = r * np.cos(t)
    #         y = r * np.sin(t)

    #         points = np.column_stack([x, y])
    #         points += rng.normal(scale=sigma, size=points.shape)

    #         return points

    #     X0 = arm(0)
    #     X1 = arm(1)

    #     X = np.vstack([X0, X1]).astype(np.float32)

    #     Y = np.array(
    #         [0] * n_objects_by_class + [1] * n_objects_by_class, dtype=np.float32
    #     ).reshape(-1, 1)

    #     return X, Y

    # X, Y = spiral_of_clouds(
    #     n_objects_by_class=1000, radius=5, n_turns=3, sigma=0.08, seed=seed, b=0.12
    # )

    # X_train_val, X_test, Y_train_val, Y_test = train_test_split(
    #     X, Y, test_size=0.20, random_state=seed, stratify=Y.ravel()
    # )

    # X_train, X_valid, Y_train, Y_valid = train_test_split(
    #     X_train_val,
    #     Y_train_val,
    #     test_size=0.25,
    #     random_state=seed,
    #     stratify=Y_train_val.ravel(),
    # )