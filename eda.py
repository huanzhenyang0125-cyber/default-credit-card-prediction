from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

root = Path.cwd()
data = pd.read_excel(root / "data/raw/default of credit card clients.xls", header=1, engine="calamine")
target = "default payment next month"
reports = root / "reports"
figures = root / "figures"
reports.mkdir(exist_ok=True)
figures.mkdir(exist_ok=True)

assert data.shape == (30000, 25)
assert set(data[target].unique()) == {0, 1}

lines = [f"Rows: {len(data)}; columns: {len(data.columns)}",
         f"Missing cells: {data.isna().sum().sum()}; duplicate IDs: {data['ID'].duplicated().sum()}",
         f"Default counts:\n{data[target].value_counts().sort_index().to_string()}",
         f"Overall default rate: {data[target].mean():.2%}"]
for column in ["SEX", "EDUCATION", "MARRIAGE", "PAY_0"]:
    grouped = data.groupby(column)[target].agg(count="size", defaults="sum", default_rate="mean")
    lines.append(f"\n{column} distribution and default rate:\n{grouped.to_string()}")

for column in ["PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
    grouped = data.groupby(column)[target].agg(count="size", defaults="sum", default_rate="mean")
    lines.append(f"\n{column} distribution and default rate:\n{grouped.to_string()}")

data["credit_group"] = pd.cut(data["LIMIT_BAL"], bins=[0, 50000, 100000, 200000, float("inf")],
                              labels=["<=50k", "50k-100k", "100k-200k", ">200k"], include_lowest=True)
credit = data.groupby("credit_group", observed=True)[target].agg(count="size", defaults="sum", default_rate="mean")
repayment = data.groupby("PAY_0")[target].agg(count="size", defaults="sum", default_rate="mean")
lines.append(f"\nCredit limit groups (NTD) and default rate:\n{credit.to_string()}")
lines.append(f"\nNumerical ranges and quantiles:\n{data[['AGE', 'LIMIT_BAL', 'BILL_AMT1', 'PAY_AMT1']].describe().to_string()}")
lines.append(f"\nNegative bill amounts by month:\n{(data[[f'BILL_AMT{i}' for i in range(1, 7)]] < 0).sum().to_string()}")
lines.append(f"\nPearson correlations (selected variables):\n{data[['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1', target]].corr().round(3).to_string()}")
output = "\n".join(lines)
(reports / "eda_results.txt").write_text(output + "\n", encoding="utf-8")
print(output)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
data[target].value_counts().sort_index().plot.bar(ax=axes[0, 0], color=["#4b7b91", "#d46b52"], rot=0)
axes[0, 0].set(title="Next-month default counts", xlabel="Default (0=no, 1=yes)", ylabel="Customers")
credit["default_rate"].mul(100).plot.bar(ax=axes[0, 1], color="#4b7b91", rot=0)
axes[0, 1].set(title="Default rate by credit limit", xlabel="Credit limit (NTD)", ylabel="Default rate (%)")
repayment["default_rate"].mul(100).plot.bar(ax=axes[1, 0], color="#d46b52", rot=0)
axes[1, 0].set(title="Default rate by September repayment code", xlabel="PAY_0 code (small groups may vary)", ylabel="Default rate (%)")
data["AGE"].plot.hist(ax=axes[1, 1], bins=20, color="#7b9f82", edgecolor="white")
axes[1, 1].set(title="Customer age distribution", xlabel="Age (years)", ylabel="Customers")
fig.savefig(figures / "eda_overview.png", dpi=150)
plt.close(fig)
print("\nSaved reports/eda_results.txt and figures/eda_overview.png")
