# Local and Streamlit Community Cloud deployment

## Local

Use Python 3.12, install `requirements.txt`, and run `python -m streamlit run app.py` from the project root. On this Windows checkout, `Open dashboard.cmd` uses the project environment and opens port 8501. The VS Code Run and Debug configuration uses port 8502.

Check the research workflow with `python tests/test_model_reproducibility.py` and `python tests/test_workspace.py`.

## Cloud preparation

1. Review the files and repository visibility before uploading. Include `app.py`, `risk_model.py`, `styles/`, `.streamlit/config.toml`, `requirements.txt`, `data/raw/default of credit card clients.xls`, and the `reports/` and `figures/` assets referenced by the app. Keep credentials, `.env`, `dashboard.log`, `.venv` and `.streamlit/secrets.toml` out of Git.
2. Place the project in a GitHub repository. No repository was created or published by this step.
3. In Streamlit Community Cloud select the repository, branch, and entrypoint `app.py`. Set Python 3.12 in advanced settings. Follow the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
4. After deployment, verify installation, each page, synthetic prediction, resource usage, and logs in the actual cloud environment. The current tests were run on local Windows only; cloud compatibility has not been verified.

The dashboard is an educational demonstration and must not be used for real credit decisions. The public display should use aggregates and synthetic examples only. No secrets are required.
