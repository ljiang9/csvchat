#!/usr/bin/env python3
"""csvchat — 用自然语言向 CSV 提问。

两步走，省 token 也保准确：
  1. 本地算出数据画像（列、类型、行数、数值统计、分类取值、样本行），
     只把画像 + 问题发给模型；模型要么直接回答，要么回一段"计算计划"。
  2. 若模型给了计算计划，用内置执行器在本地真实 CSV 上跑出结果，
     再把结果发给模型，生成最终的自然语言回答。

只用 Python 标准库。API Key 绝不打印到任何输出里。
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.request

VERSION = "0.1.0"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
REQUEST_TIMEOUT = 60

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class CsvChatError(Exception):
    """用户可见的错误（中文信息），不携带 API Key。"""


# ---------------------------------------------------------------- 数据画像

def load_csv(path):
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise CsvChatError(f"CSV 文件没有表头: {path}")
            rows = list(reader)
    except FileNotFoundError:
        raise CsvChatError(f"找不到 CSV 文件: {path}")
    except CsvChatError:
        raise
    except Exception as e:
        raise CsvChatError(f"读取 CSV 失败 {path}: {e}")
    if not rows:
        raise CsvChatError(f"CSV 文件没有数据行: {path}")
    return reader.fieldnames, rows


def _is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def infer_dtype(values):
    """从一列的非空值推断类型：int / float / date / string。"""
    vals = [v for v in values if v not in (None, "")]
    if not vals:
        return "string"
    if all(re.fullmatch(r"[+-]?\d+", v.strip()) for v in vals):
        return "int"
    if all(_is_number(v.strip()) for v in vals):
        return "float"
    if all(DATE_RE.match(v.strip()) for v in vals):
        return "date"
    return "string"


def build_profile(columns, rows):
    profile = {
        "row_count": len(rows),
        "columns": [],
        "numeric_stats": {},
        "categorical_values": {},
        "sample_rows": rows[:5],
    }
    for col in columns:
        vals = [r[col] for r in rows]
        dtype = infer_dtype(vals)
        profile["columns"].append({"name": col, "dtype": dtype})
        if dtype in ("int", "float"):
            nums = [float(v) for v in vals if v not in (None, "")]
            profile["numeric_stats"][col] = {
                "min": min(nums),
                "max": max(nums),
                "mean": round(sum(nums) / len(nums), 2),
            }
        else:
            freq = {}
            for v in vals:
                key = "(空)" if v in (None, "") else v
                freq[key] = freq.get(key, 0) + 1
            top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            profile["categorical_values"][col] = [
                {"value": v, "count": c} for v, c in top
            ]
    return profile


def format_profile(profile):
    L = []
    L.append("# 数据画像")
    L.append(f"- 共 {profile['row_count']} 行，{len(profile['columns'])} 列")
    cols = ", ".join(f"{c['name']}({c['dtype']})" for c in profile["columns"])
    L.append(f"- 列: {cols}")
    if profile["numeric_stats"]:
        L.append("")
        L.append("## 数值列统计")
        for col, s in profile["numeric_stats"].items():
            L.append(f"- {col}: 最小 {s['min']}, 最大 {s['max']}, 平均 {s['mean']}")
    if profile["categorical_values"]:
        L.append("")
        L.append("## 分类列取值 (top 5)")
        for col, items in profile["categorical_values"].items():
            vals = ", ".join(f"{i['value']}({i['count']})" for i in items)
            L.append(f"- {col}: {vals}")
    L.append("")
    L.append("## 样本行 (前 5 行)")
    cols = [c["name"] for c in profile["columns"]]
    L.append("| " + " | ".join(cols) + " |")
    L.append("| " + " | ".join("---" for _ in cols) + " |")
    for r in profile["sample_rows"]:
        L.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(L)


# ---------------------------------------------------------------- 计算计划执行器

PLAN_OPS_DOC = """filter 列=值 | filter 列!=值 | filter 列>值 | filter 列<值 | filter 列>=值 | filter 列<=值
group 列
sum 列 / mean 列 / count
sort 列 [asc|desc]
limit 数字

聚合结果列命名为 sum_列 / mean_列 / count，例如:
  filter region=华东 | group product | sum revenue | sort sum_revenue desc | limit 3"""


def _to_num(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_plan(plan_text):
    """把计划文本解析成 stage 列表。失败抛 CsvChatError。"""
    stages = []
    for raw in plan_text.split("|"):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        op = parts[0].lower()
        if op == "filter":
            m = re.fullmatch(r"(.+?)(!=|>=|<=|=|>|<)(.+)", raw[len("filter"):].strip())
            if not m:
                raise CsvChatError(f"计划解析失败: filter 语法错误: {raw!r}")
            col, cmp, val = m.group(1).strip(), m.group(2), m.group(3).strip()
            if not col:
                raise CsvChatError(f"计划解析失败: filter 缺少列名: {raw!r}")
            stages.append(("filter", col, cmp, val))
        elif op == "group":
            if len(parts) != 2:
                raise CsvChatError(f"计划解析失败: group 需要 1 个列名: {raw!r}")
            stages.append(("group", parts[1]))
        elif op in ("sum", "mean"):
            if len(parts) != 2:
                raise CsvChatError(f"计划解析失败: {op} 需要 1 个列名: {raw!r}")
            stages.append((op, parts[1]))
        elif op == "count":
            stages.append(("count",))
        elif op == "sort":
            if len(parts) not in (2, 3):
                raise CsvChatError(f"计划解析失败: sort 语法为 sort 列 [asc|desc]: {raw!r}")
            direction = parts[2].lower() if len(parts) == 3 else "asc"
            if direction not in ("asc", "desc"):
                raise CsvChatError(f"计划解析失败: sort 方向只能是 asc/desc: {raw!r}")
            stages.append(("sort", parts[1], direction))
        elif op == "limit":
            if len(parts) != 2 or not parts[1].isdigit():
                raise CsvChatError(f"计划解析失败: limit 需要一个正整数: {raw!r}")
            stages.append(("limit", int(parts[1])))
        else:
            raise CsvChatError(
                f"计划解析失败: 不支持的操作 {op!r}。支持的操作:\n{PLAN_OPS_DOC}")
    if not stages:
        raise CsvChatError("计划解析失败: 计划为空")
    return stages


def _cmp_match(cell, cmp, val):
    cn, vn = _to_num(cell), _to_num(val)
    if cn is not None and vn is not None:
        a, b = cn, vn
    else:
        a, b = str(cell), str(val)
    return {"=": a == b, "!=": a != b, ">": a > b, "<": a < b,
            ">=": a >= b, "<=": a <= b}[cmp]


def execute_plan(stages, columns, rows):
    """在本地真实数据上执行计划，返回结果行（list[dict]）。"""
    data = [dict(r) for r in rows]
    groups = None  # 当前分组列，未分组时为 None

    def check_col(col):
        if col not in columns:
            raise CsvChatError(f"计划执行失败: CSV 里没有这一列: {col!r}")

    for st in stages:
        op = st[0]
        if op == "filter":
            _, col, cmp, val = st
            check_col(col)
            data = [r for r in data if _cmp_match(r.get(col, ""), cmp, val)]
            groups = None
        elif op == "group":
            check_col(st[1])
            groups = st[1]
        elif op in ("sum", "mean"):
            col = st[1]
            check_col(col)
            out = []
            if groups:
                seen, order = {}, []
                for r in data:
                    k = r.get(groups, "")
                    if k not in seen:
                        seen[k] = []
                        order.append(k)
                    seen[k].append(r)
                for k in order:
                    nums = [_to_num(r.get(col, "")) for r in seen[k]]
                    nums = [n for n in nums if n is not None]
                    if not nums:
                        raise CsvChatError(f"计划执行失败: 列 {col!r} 没有可聚合的数字")
                    v = sum(nums) if op == "sum" else sum(nums) / len(nums)
                    out.append({groups: k, f"{op}_{col}": round(v, 2)})
            else:
                nums = [_to_num(r.get(col, "")) for r in data]
                nums = [n for n in nums if n is not None]
                if not nums:
                    raise CsvChatError(f"计划执行失败: 列 {col!r} 没有可聚合的数字")
                v = sum(nums) if op == "sum" else sum(nums) / len(nums)
                out.append({f"{op}_{col}": round(v, 2)})
            data = out
            groups = None
        elif op == "count":
            out = []
            if groups:
                seen, order = {}, []
                for r in data:
                    k = r.get(groups, "")
                    if k not in seen:
                        seen[k] = 0
                        order.append(k)
                    seen[k] += 1
                for k in order:
                    out.append({groups: k, "count": seen[k]})
            else:
                out.append({"count": len(data)})
            data = out
            groups = None
        elif op == "sort":
            _, col, direction = st
            if data and col not in data[0]:
                raise CsvChatError(f"计划执行失败: 结果里没有这一列: {col!r}")
            numvals = [_to_num(r.get(col)) for r in data]
            if all(n is not None for n in numvals):
                key = lambda r: _to_num(r.get(col))
            else:
                key = lambda r: str(r.get(col, ""))
            data = sorted(data, key=key, reverse=(direction == "desc"))
        elif op == "limit":
            data = data[:st[1]]
    return data


def format_result_table(rows):
    if not rows:
        return "(计算结果为空)"
    cols = list(rows[0].keys())
    L = ["| " + " | ".join(cols) + " |",
         "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        L.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(L)


# ---------------------------------------------------------------- 模型调用

STEP1_SYSTEM = """你是一个数据问答助手。用户会给你一份 CSV 的数据画像和一个问题。
画像包含列名、类型、行数、数值统计、分类取值和 5 个样本行。

请严格用 JSON 回复，只用以下两种格式之一，不要输出其他内容：

1. 如果问题能直接从画像回答：
{"direct": "用中文写出的答案"}

2. 如果需要对全量数据做精确计算，给出"计算计划"（用竖线分隔的流水线）：
{"plan": "filter region=华东 | group product | sum revenue | sort sum_revenue desc | limit 3"}

计算计划支持的操作：
filter 列=值 | filter 列!=值 | filter 列>值 | filter 列<值 | filter 列>=值 | filter 列<=值
group 列
sum 列 / mean 列 / count
sort 列 [asc|desc]
limit 数字
聚合结果列命名为 sum_列 / mean_列 / count，可直接在 sort 里引用。

规则：
- 只用画像里真实存在的列名，不要编造。
- filter 的值用画像里的真实取值（如 region=华东）。
- 不要假设画像之外的信息；需要全量数据时一律走 plan。
"""

STEP2_SYSTEM = """你是一个数据问答助手。用户的问题已经通过计算计划在真实数据上算出结果。
请根据计算结果用简体中文给出简洁准确的最终回答，只说结论和关键数字，不要复述计划语法。"""


def call_api(base_url, api_key, model, system, user, temperature=0.2):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise CsvChatError(f"模型 API 返回错误 HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise CsvChatError(f"网络请求失败: {e.reason}")
    except json.JSONDecodeError:
        raise CsvChatError("模型 API 返回的不是合法 JSON")
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise CsvChatError("模型 API 返回结构异常，取不到回答内容")


def parse_step1_reply(text):
    """解析 step1 的 JSON 回复，返回 ("direct", 答案) 或 ("plan", 计划)。"""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        raise CsvChatError(f"模型第一步没有返回合法 JSON，无法继续。原文:\n{text[:300]}")
    if isinstance(obj, dict) and "direct" in obj:
        return "direct", str(obj["direct"])
    if isinstance(obj, dict) and "plan" in obj:
        return "plan", str(obj["plan"])
    raise CsvChatError(f"模型第一步返回的 JSON 里没有 direct 或 plan 字段。原文:\n{text[:300]}")


def get_api_key():
    key = os.environ.get("CSVCHAT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise CsvChatError(
            "未找到 API Key。请先设置环境变量后再重试:\n"
            "  export OPENAI_API_KEY=你的key   (或 CSVCHAT_API_KEY)")
    return key


# ---------------------------------------------------------------- 主流程

def build_step1_prompt(profile_text, question):
    return f"{profile_text}\n\n## 用户问题\n{question}"


def build_step2_prompt(question, plan_text, result_table):
    return (f"## 用户问题\n{question}\n\n"
            f"## 已执行的计算计划\n{plan_text}\n\n"
            f"## 计算结果\n{result_table}\n\n"
            "请基于以上计算结果回答用户问题。")


def run(csv_path, question, model, base_url, dry_run=False):
    columns, rows = load_csv(csv_path)
    profile = build_profile(columns, rows)
    profile_text = format_profile(profile)

    step1_user = build_step1_prompt(profile_text, question)
    if dry_run:
        print("===== STEP 1 PROMPT (发给模型) =====")
        print("【system】")
        print(STEP1_SYSTEM)
        print("【user】")
        print(step1_user)
        print()
        print("===== STEP 2 PROMPT (若模型返回计算计划，执行后再发) =====")
        print("【system】")
        print(STEP2_SYSTEM)
        print("【user】")
        print(build_step2_prompt(question, "<模型返回的计算计划>",
                                 "<本地执行计划后得到的计算结果表>"))
        return {"dry_run": True}

    api_key = get_api_key()
    reply = call_api(base_url, api_key, model, STEP1_SYSTEM, step1_user)
    kind, payload = parse_step1_reply(reply)

    if kind == "direct":
        return {"answer": payload, "plan": None, "result_rows": []}

    stages = parse_plan(payload)
    result_rows = execute_plan(stages, columns, rows)
    result_table = format_result_table(result_rows)
    step2_user = build_step2_prompt(question, payload, result_table)
    answer = call_api(base_url, api_key, model, STEP2_SYSTEM, step2_user)
    return {"answer": answer.strip(), "plan": payload, "result_rows": result_rows}


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="csvchat",
        description="用自然语言向 CSV 提问：本地画像 + 模型推理 + 本地精确计算。")
    ap.add_argument("csv", nargs="?", help="CSV 文件路径")
    ap.add_argument("question", nargs="?", help="要问的问题")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"模型名 (默认 {DEFAULT_MODEL})")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI 兼容接口地址 (默认读 $OPENAI_BASE_URL，否则官方地址)")
    ap.add_argument("--profile", action="store_true", help="只打印数据画像，不提问")
    ap.add_argument("--dry-run", action="store_true", help="只打印两步 prompt，不调网络")
    ap.add_argument("--json", action="store_true", help="最终答案用 JSON 输出")
    ap.add_argument("--version", action="version", version=f"csvchat {VERSION}")
    args = ap.parse_args(argv)

    try:
        if args.profile:
            if not args.csv:
                raise CsvChatError("--profile 需要指定 CSV 文件路径")
            columns, rows = load_csv(args.csv)
            print(format_profile(build_profile(columns, rows)))
            return 0
        if not args.csv or not args.question:
            ap.print_usage(sys.stderr)
            print("error: 需要 CSV 文件路径和问题，例如:\n"
                  '  csvchat data.csv "华东地区哪个产品收入最高？"',
                  file=sys.stderr)
            return 2
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        model = args.model or os.environ.get("CSVCHAT_MODEL") or DEFAULT_MODEL
        out = run(args.csv, args.question, model, base_url, dry_run=args.dry_run)
        if args.json and not out.get("dry_run"):
            print(json.dumps({"answer": out["answer"], "plan": out["plan"],
                              "result_rows": out["result_rows"]},
                             ensure_ascii=False, indent=2))
        elif not out.get("dry_run"):
            print(out["answer"])
        return 0
    except CsvChatError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
