# 任务背景：COVID-19 住院预测（论文协议接口）

供研究想法系统 S15 使用。接口与沙箱环境按 2026-10-09 恢复的论文协议描述：原评分 wrapper、
`fit_and_predict_fn` 接口、config_list 外部选优、隔离容器执行。以下英文正文为给模型的任务背景。

---

I am developing new methods for the main US benchmark for COVID-19 forecasting, the COVID-19 Forecast Hub (CovidHub), a large collaborative effort coordinated by the CDC that receives weekly forecasts from dozens of expert-led teams. Briefly: forecasters must predict new COVID-19-related hospital admissions across 52 US states and territories for four weekly horizons over 23 specified quantiles. Submissions are evaluated using the weighted interval score (WIS), which rewards both accuracy and well-calibrated uncertainty.

The key problem is to develop a method that defines exactly this callable:

    fit_and_predict_fn(train_x, train_y, test_x, config)

All arguments are supplied by the evaluation harness. train_x is a pandas DataFrame of the observed history up to seven days before the forecast reference date, with columns location (FIPS), location_name, population, and target_end_date (weekly, ending Saturdays); train_y is the corresponding pandas Series of observed weekly admissions counts (missing values are imputed as zero). test_x holds the rows to forecast: one row per location and target week, with target weeks ending one to four weeks after the last observed week. config carries one option dictionary chosen from the config_list that your code returns alongside the function; the harness itself scores every configuration on six validation reference dates and selects the best one, so the function must not do any internal cross-validation. The function must return a pandas DataFrame with exactly one row per test_x row and 23 quantile columns named quantile_0.01, quantile_0.025, quantile_0.05, quantile_0.1 through quantile_0.9, quantile_0.95, quantile_0.975, quantile_0.99; forecasts must be finite and meaningful across quantiles.

The following principles should be obeyed when choosing models:

* **Probabilistic Calibration**: Prioritize techniques whose full predictive quantiles are well calibrated, since the WIS penalizes both bias and dispersion errors across all 23 quantile levels.
* **Uncertainty Awareness**: Ensure predictive uncertainty grows sensibly with forecast horizon and reflects empirical epidemic volatility rather than fixed spreads.
* **Robustness to Data Pathologies**: Handle missing histories, zero-admission locations, reporting revisions and holiday effects without failing entire reference dates; a whole candidate is rejected if any reference date fails.
* **Scalability and Efficiency**: The evaluation runs each candidate inside an isolated container with two CPU cores and 2 GiB RAM (8 GiB when a GPU is attached), with roughly twenty minutes of total execution per candidate across all configurations and reference dates. Available libraries are numpy, pandas, scipy, scikit-learn, lightgbm, statsmodels and PyTorch (CPU or GPU), so methods should stay computationally efficient and avoid out-of-memory errors.
* **Constraint Adherence**: The implementation must not do any cross-validation, label lookup or internal tuning inside the function; random generators should be seeded deterministically; only the supplied training history may be used, with no external data sources.

This task aims to develop a SUPERHUMAN METHOD for solving this problem.

Please give me 10 highly novel and creative ideas with detailed implementation notes for the set of methods I should explore for solving this task. I aim to create the best method for solving this problem, preferably creating the best ever method.
