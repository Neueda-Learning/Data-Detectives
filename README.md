# Data-Detectives

信用卡客户违约风险分析项目（从原始 Excel 到清洗、探索分析、Power BI 展示数据）。

## 1. 生成结果（你最终会得到什么）

项目运行后会产出以下结果文件：

- [credit_data_clean.csv](credit_data_clean.csv)
  - 清洗后的主数据（30,000 行，含清洗与衍生特征），用于后续分析与建模。
- [data_diary.md](data_diary.md)
  - 数据清洗决策日志，记录每个处理动作的原因，便于答辩与复现。
- [charts](charts)
  - 探索分析阶段导出的图表 PNG，用于讲故事和汇报。
- [powerbi_main.csv](powerbi_main.csv)
  - Power BI 主表（每行一个客户），适合维度对比、分群、风险画像。
- [powerbi_monthly.csv](powerbi_monthly.csv)
  - Power BI 明细月度表（每行=客户 x 月份），适合趋势线与时间序列图。
- [summary_reference.csv](summary_reference.csv)
  - 关键口径对照表（如各分组违约率），用于校验 Power BI 结果是否一致。

## 2. 图片（charts）是干什么的

图表按 A-E 五条分析主线组织：

- A 人口特征（区分度较弱）
  - [a1_sex.png](charts/a1_sex.png): 按性别看违约率
  - [a2_education.png](charts/a2_education.png): 按学历看违约率
  - [a3_marriage.png](charts/a3_marriage.png): 按婚姻看违约率
  - [a4_age_band.png](charts/a4_age_band.png): 按年龄段看违约率

- B 信用暴露（区分度中等）
  - [b1_limit_band.png](charts/b1_limit_band.png): 按信用额度分位看违约率
  - [b2_utilization_band.png](charts/b2_utilization_band.png): 按额度使用率分位看违约率
  - [b3_utilization_boxplot.png](charts/b3_utilization_boxplot.png): 违约与未违约客户的使用率分布对比

- C 还款行为（区分度最强）
  - [c1_ever_delayed.png](charts/c1_ever_delayed.png): 是否曾逾期 vs 违约率
  - [c2_months_delayed.png](charts/c2_months_delayed.png): 逾期月数 vs 违约率
  - [c3_pay_month_correlation.png](charts/c3_pay_month_correlation.png): 各月份还款状态与违约相关性

- D 时间趋势（行为变化）
  - [d1_trend_bill_payment.png](charts/d1_trend_bill_payment.png): 账单金额与还款金额随时间变化
  - [d2_payment_trend_band.png](charts/d2_payment_trend_band.png): 还款趋势分位 vs 违约率

- E 综合风险（可落地分群）
  - [e1_feature_ranking.png](charts/e1_feature_ranking.png): 特征与违约相关性强弱排序
  - [e2_risk_segment.png](charts/e2_risk_segment.png): 三档风险分群对应违约率

## 3. 文件用途总览

- [default of credit card clients.xls](default%20of%20credit%20card%20clients.xls): 原始数据
- [clean_data.py](clean_data.py): 清洗与特征工程，生成 clean 数据与 diary
- [explore_credit_data.py](explore_credit_data.py): 探索分析，输出 charts
- [prepare_powerbi_data.py](prepare_powerbi_data.py): 准备 Power BI 三张输入表
- [presentation_script.md](presentation_script.md): 课堂/答辩讲稿

## 4. 如何使用（推荐顺序）

### 第 1 步：进入项目目录

    cd /Users/zxx/CODE/Data-Detectives

### 第 2 步：激活环境并安装依赖

如果你使用项目内环境：

    conda activate /Users/zxx/CODE/Data-Detectives/.conda

安装依赖：

    /Users/zxx/CODE/Data-Detectives/.conda/bin/python -m pip install numpy pandas matplotlib xlrd

说明：
- 读取 .xls 原始文件需要 xlrd。
- 如果报 No module named xxx，就在当前解释器里补装对应包。

### 第 3 步：运行清洗脚本

    /Users/zxx/CODE/Data-Detectives/.conda/bin/python clean_data.py

产出：
- [credit_data_clean.csv](credit_data_clean.csv)
- [data_diary.md](data_diary.md)

### 第 4 步：运行探索分析脚本

    /Users/zxx/CODE/Data-Detectives/.conda/bin/python explore_credit_data.py

产出：
- [charts](charts) 下全部图片

### 第 5 步：生成 Power BI 输入文件

    /Users/zxx/CODE/Data-Detectives/.conda/bin/python prepare_powerbi_data.py

产出：
- [powerbi_main.csv](powerbi_main.csv)
- [powerbi_monthly.csv](powerbi_monthly.csv)
- [summary_reference.csv](summary_reference.csv)

### 第 6 步：导入 Power BI

建议导入三张表：
- [powerbi_main.csv](powerbi_main.csv)
- [powerbi_monthly.csv](powerbi_monthly.csv)
- [summary_reference.csv](summary_reference.csv)

关系设置：
- powerbi_main.ClientID (1) -> powerbi_monthly.ClientID (*)

## 5. 常见问题

- 问：报错 No module named numpy/pandas/xlrd 为什么？
  - 答：你当前运行脚本的 Python 解释器不是安装依赖的那个解释器。请用同一个解释器执行安装和运行（例如都使用 /Users/zxx/CODE/Data-Detectives/.conda/bin/python）。

- 问：为什么不删除异常编码行（如 EDUCATION 0/5/6）？
  - 答：项目策略是尽量保留预测信号，异常编码按业务逻辑并入 Others，详见 [data_diary.md](data_diary.md)。
