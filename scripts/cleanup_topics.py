#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topic_queue.json 机械化清理脚本：
1. 标准化字段顺序为 id, title, slug, tags, brief, depth_hint, status
2. 去掉「结构树缺口补齐与工程化落地」标题后缀（288条）
3. 删除 ID 8276（标题为约束规则文字的异常条目）
"""

import json
import re

STANDARD_ORDER = ["id", "title", "slug", "tags", "brief", "depth_hint", "status"]
STRUCT_SUFFIX = "：结构树缺口补齐与工程化落地"
DELETE_IDS = {8276}  # 标题是规则文字的异常条目


def fix_field_order(item: dict) -> dict:
    return {k: item[k] for k in STANDARD_ORDER if k in item}


def clean_title(title: str) -> str:
    if STRUCT_SUFFIX in title:
        return title.split(STRUCT_SUFFIX)[0].strip()
    return title


def fix_slug_for_cleaned_title(item: dict) -> dict:
    """如果标题被清理了，检查 slug 是否也需要去掉后缀。"""
    slug = item.get("slug", "")
    # 常见的结构树缺口后缀 slug 模式
    if slug.endswith("-jiegou-shu-que-kou-bu-qi-yu-gong-cheng-hua-luo-di"):
        item["slug"] = slug.replace(
            "-jiegou-shu-que-kou-bu-qi-yu-gong-cheng-hua-luo-di", ""
        )
    elif "-jiegoushu-" in slug:
        # 清理其他可能的变体
        pass
    return item


def main():
    input_file = "topic_queue.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"加载条目数: {len(data)}")

    cleaned = []
    deleted_count = 0
    title_fixed_count = 0
    field_fixed_count = 0

    for item in data:
        # 1. 删除异常条目
        if item.get("id") in DELETE_IDS:
            deleted_count += 1
            print(f"删除异常条目 ID {item['id']}: {item['title'][:60]}")
            continue

        # 2. 清理标题后缀
        original_title = item.get("title", "")
        cleaned_title = clean_title(original_title)
        if cleaned_title != original_title:
            item["title"] = cleaned_title
            title_fixed_count += 1

        # 3. 标准化字段顺序
        keys = list(item.keys())
        if keys != STANDARD_ORDER and keys[:len(STANDARD_ORDER)] != STANDARD_ORDER:
            field_fixed_count += 1

        cleaned.append(fix_field_order(item))

    print(f"删除异常条目: {deleted_count}")
    print(f"标题后缀清理: {title_fixed_count}")
    print(f"字段顺序修正: {field_fixed_count}")
    print(f"输出条目数: {len(cleaned)}")

    # 验证 slug 唯一性
    slugs = [item["slug"] for item in cleaned]
    dup_slugs = len(slugs) - len(set(slugs))
    if dup_slugs > 0:
        from collections import Counter
        dup_list = [s for s, c in Counter(slugs).items() if c > 1]
        print(f"警告：仍有 {dup_slugs} 个重复 slug，前5个：{dup_list[:5]}")
    else:
        print("slug 唯一性验证：通过（无重复）")

    # 验证标题唯一性
    titles = [item["title"] for item in cleaned]
    dup_titles = len(titles) - len(set(titles))
    if dup_titles > 0:
        from collections import Counter
        dup_list = [t for t, c in Counter(titles).items() if c > 1]
        print(f"警告：仍有 {dup_titles} 个重复标题，前5个：{dup_list[:5]}")
    else:
        print("标题唯一性验证：通过（无重复）")

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"\n已保存到 {input_file}")


if __name__ == "__main__":
    main()
