import pandas as pd
import joblib

def main():
    df = pd.read_csv("dataset.csv")

    X, y = split_xy(df, cfg)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    pipe = build_pipeline(cfg)

    pipe.fit(X_train, y_train)

    metrics = evaluate_binary(pipe, X_valid, y_valid)
    logger.info("Metrics: %s", json.dumps(metrics, ensure_ascii=False))

    out_path = MODELS_DIR / cfg.model_filename
    joblib.dump(pipe, out_path)
    logger.info("Saved model to: %s", out_path)

if __name__ == "__main__":
    main()
