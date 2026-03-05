#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix topic_queue.json:
1. Standardize field ordering to: id, title, slug, tags, brief, depth_hint, status
2. Enrich skeleton briefs (< 50 chars) with substantive content
"""

import json
import sys

ENRICHED_BRIEFS = {}

def load_enrichments():
    global ENRICHED_BRIEFS
    # Will be populated by subsequent script parts
    pass

def fix_field_order(item):
    """Reorder fields to standard: id, title, slug, tags, brief, depth_hint, status"""
    return {
        "id": item["id"],
        "title": item["title"],
        "slug": item["slug"],
        "tags": item["tags"],
        "brief": item.get("brief", ""),
        "depth_hint": item.get("depth_hint", ""),
        "status": item.get("status", "pending")
    }

def main():
    input_file = "topic_queue.json"
    output_file = "topic_queue.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items")

    # Load enrichment data
    try:
        with open("enrichments.json", "r", encoding="utf-8") as f:
            enrichments = json.load(f)
        print(f"Loaded {len(enrichments)} enrichments")
    except FileNotFoundError:
        enrichments = {}
        print("No enrichments file found, only fixing field order")

    fixed = []
    enriched_count = 0
    field_fixed_count = 0

    for item in data:
        # Check if field order needs fixing
        keys = list(item.keys())
        if keys[0] != "id":
            field_fixed_count += 1

        # Apply enrichment if available
        item_id = str(item["id"])
        if item_id in enrichments:
            brief_data = enrichments[item_id]
            if "brief" in brief_data and len(brief_data["brief"]) > len(item.get("brief", "")):
                item["brief"] = brief_data["brief"]
                enriched_count += 1
            if "depth_hint" in brief_data and brief_data["depth_hint"]:
                item["depth_hint"] = brief_data["depth_hint"]

        # Fix field order
        fixed.append(fix_field_order(item))

    print(f"Fixed field order for {field_fixed_count} items")
    print(f"Enriched {enriched_count} briefs")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(fixed)} items to {output_file}")

if __name__ == "__main__":
    main()
