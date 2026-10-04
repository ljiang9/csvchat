# csvchat

**用自然语言向 CSV 提问，不用 pandas、不用 notebook。**

```bash
export OPENAI_API_KEY=你的key

python -m csvchat examples/sales.csv "华东地区哪个产品收入最高？"
# → 华东地区收入最高的产品是机械键盘，总收入 116028.68 元。
```

## 怎么做到又便宜又准：两步走

直接把整个 CSV 丢给模型又贵又容易幻觉。csvchat 分两步：

1. **本地画像**：只用标准库在本地算出数据画像（列名、类型、行数、数值 min/max/mean、分类 top5、5 个样本行），**只把画像 + 问题发给模型**，原始数据不出本机。
2. **计划执行**：模型要么直接回答（画像里能答的），要么回一段"计算计划"（见下）；csvchat 用内置执行器在**本地真实 CSV** 上跑出精确结果，再让模型组织成自然语言。

## 安装

零依赖，Python 3.10+ 自带一切：

```bash
git clone https://github.com/ljiang9/csvchat.git
cd csvchat
export OPENAI_API_KEY=你的key   # 或 CSVCHAT_API_KEY
```

也支持任何 OpenAI 兼容接口：`export OPENAI_BASE_URL=https://...`

## 用法

```bash
# 提问
python -m csvchat data.csv "华东地区哪个产品收入最高？"

# 只看数据画像（不调 API，免费）
python -m csvchat data.csv --profile

# 预览两步 prompt（不调 API，看会发什么给模型）
python -m csvchat data.csv "平均客单价多少？" --dry-run

# 机器可读输出
python -m csvchat data.csv "哪个地区销量最大？" --json

# 指定模型
python -m csvchat data.csv "问题" --model gpt-4o-mini
```

## 计算计划语言

模型返回的 plan 是用 `|` 分隔的流水线，操作从左到右执行：

| 操作 | 说明 | 示例 |
|---|---|---|
| `filter 列=值` | 过滤行（也支持 `!=` `>` `<` `>=` `<=`，数字按数字比） | `filter region=华东` |
| `group 列` | 按列分组 | `group product` |
| `sum 列` | 求和（分组后按组求和） | `sum revenue` |
| `mean 列` | 求平均 | `mean revenue` |
| `count` | 计数（分组后按组计数） | `count` |
| `sort 列 [asc\|desc]` | 按列排序，默认升序 | `sort sum_revenue desc` |
| `limit 数字` | 取前 N 行 | `limit 3` |

聚合结果列固定命名为 `sum_列` / `mean_列` / `count`，可直接在 `sort` 里引用。

完整例子：

```
filter region=华东 | group product | sum revenue | sort sum_revenue desc | limit 3
```

含义：只留华东的行 → 按产品分组 → 每组 revenue 求和 → 按总和降序 → 取前 3。

计划写错了（未知操作、列名不存在）会直接报中文错、退出码 1，不会把坏计划发给模型。

## 隐私

- 原始 CSV 行数据**永远不会**发给模型，只有画像（统计 + 5 个样本行）和计算结果表会发送。
- API Key 只从环境变量读取，任何输出里都不会出现。

## 已知局限

- 计划语言是刻意做小的：没有 join、没有多列 group by、没有复杂表达式，超纲问题模型只能尽力拆。
- 模型必须遵守 JSON 回复格式（`{"direct": ...}` / `{"plan": ...}`），不遵守时会报中文错退出。
- 类型推断是启发式的（int/float/date/string），脏数据可能推错；先用 `--profile` 看一眼画像再问。

## License

MIT
