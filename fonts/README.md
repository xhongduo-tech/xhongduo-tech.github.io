# Self hosted Latin web font subsets

Modified, Unicode-subsetted derivatives of the official fonts imported on 2026-11-16. Glyph designs and complete variable axes are retained. Each family is split into disjoint Latin/core and Extended packages. Chinese uses system fallback fonts; no proprietary Anthropic fonts or logos are included.

## License and derivative names

All derivatives remain under SIL OFL 1.1. Original copyright, author, license and trademark metadata are retained in each font. Original OFL.txt files remain in source-serif-4/, inter/ and jetbrains-mono/; retain these notices with redistribution.

Source Serif's Reserved Font Name is `Source`. The supplied Inter and JetBrains Mono license headers do not specify a Reserved Font Name. All three derivatives have custom internal family, full, unique, PostScript, variation PostScript prefix and named-instance PostScript names. Copyright/trademark acknowledgments intentionally retain upstream names and do not endorse the derivatives.

CSS families are `XHD Serif`, `XHD Sans` and `XHD Mono`. Latin and Extended files share a family but have distinct full/PostScript names. These are modified derivatives, not unmodified upstream releases.

## Delivered files

| File | CSS family | Bytes | Glyphs | Unicode cmap entries | SHA256 |
| --- | --- | ---: | ---: | ---: | --- |
| XHDSerif-Latin.woff2 | XHD Serif | 130,052 | 383 | 267 | `eb52363f8af6d0556e22c08c5672d67709b1cb4ba7461ecf3673ce530573a5b7` |
| XHDSerif-Extended.woff2 | XHD Serif | 132,772 | 548 | 430 | `9c0dfba0d1dc37f3b95d12658cf635d59491dfaf927e830b85de6608c0946fd9` |
| XHDSans-Latin.woff2 | XHD Sans | 99,660 | 804 | 374 | `686224ee7c322dd4b971751a0e58e661d55a517ca34403ada110010d6f1b30c2` |
| XHDSans-Extended.woff2 | XHD Sans | 121,380 | 950 | 798 | `7648e8ee84c78f89e60b01d4cff804c6d9bf3f798c3e555b7ab0b2890e168916` |
| XHDMono-Latin.woff2 | XHD Mono | 45,560 | 524 | 293 | `7c203589ed1ebc44cf1c370b5e65472d6ba9e214d0929fec24e43c268fa812a7` |
| XHDMono-Extended.woff2 | XHD Mono | 32,740 | 605 | 495 | `27e81b64e0bb474199027acd3f4c269bd771da08752b9af557e066a27effc6b9` |

Core payload: 275,272 bytes; Extended payload: 286,892 bytes; combined: 562,164 bytes. Original unsplit payload: 895,044 bytes. Core is 69.2% smaller than that original payload. The current homepage's Latin characters are in core; actual requests and loading performance require browser measurement with matching unicode-range declarations.

## Exact shared CSS ranges

Use identical family/style/axis declarations for the two faces, their corresponding files, and the following disjoint ranges. Each cmap contains only the original font's supported characters within its assigned range; subsetting does not create missing glyphs.

Latin/core:

```css
unicode-range: U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0300-036F,U+2000-206F,U+20AC,U+2113,U+2190-21FF,U+2212,U+FEFF,U+FFFD;
```

Extended:

```css
unicode-range: U+0100-0130,U+0132-0151,U+0154-024F,U+0370-03FF,U+1E00-1EFF,U+2070-209F,U+20A0-20AB,U+20AD-20CF,U+2100-2112,U+2114-214F,U+2200-2211,U+2213-22FF;
```

Core includes ASCII/Latin-1, selected common Latin/modifier characters, combining accents, general punctuation, arrows, Euro and mathematical minus. Extended supplies the remaining defined web scope: Latin extensions through U+024F, Greek U+0370–03FF, Latin Additional U+1E00–1EFF, superscripts, currencies, letterlike symbols and basic math. This bounded web scope is not all historical Latin blocks or every mathematical symbol. Missing glyphs continue to use CSS fallbacks.

## Processing and validation

- Existing fontTools 4.63.0 with WOFF2 support; no project dependency changes.
- Unicode/script subsetting and internal renaming; no variable-axis instancing or outline redesign.
- Standard default layout features and hinting retained. Unused optional features and their glyph closure are not forced with `layout_features=*`.
- `retain_gids=False`, `notdef_outline=False`, `glyph_names=False`; original name/license fields preserved; `recalcTimestamp=False`.
- A redundant U+27DC cmap alias retained by subsetting in Mono Extended was removed to enforce the exact declared range; its shared glyph was not redesigned.
- All six serialized files were reopened. Each cmap equals the assigned range intersected with its original cmap. Core/Extended sets are disjoint and their union covers the defined supported scope.
- Complete fvar axis bounds/defaults and named-instance coordinates/style labels compare unchanged. Existing fvar/gvar/avar/STAT/HVAR/MVAR tables remain present.
- Family/full/unique/PS/prefix and named-instance PS names were checked for custom names across language/platform records. Original copyright, author and license fields compare unchanged.
- A fixture covering the selected brand/H1, accents, digits, punctuation, arrows and code/math symbols was checked against original supported coverage. Unsupported upstream glyphs are not claimed as new coverage.
- Metadata and coverage checks do not replace browser typography and before/after performance checks.

## Axes retained

| Family | Normal/Roman variable axes |
| --- | --- |
| XHD Serif | wght 200–900, default 400; opsz 8–60, default 20 |
| XHD Sans | wght 100–900, default 400; opsz 14–32, default 14 |
| XHD Mono | wght 100–800, default 400 |

No italic font was added. The original unsubsetted WOFF2 files were removed from the public asset directory. Their fixed upstream sources and original hashes remain below for provenance and reconstruction.

## Original upstream Source Serif 4 Variable

- Repository: https://github.com/adobe-fonts/source-serif
- Fixed commit: `80d3f8894c09c937bebfa9011247d2e1c79fd6f4`
- Official source: https://raw.githubusercontent.com/adobe-fonts/source-serif/80d3f8894c09c937bebfa9011247d2e1c79fd6f4/WOFF2/VAR/SourceSerif4Variable-Roman.ttf.woff2
- Original source SHA256: `940a76eda1388de39d38c8e7a79bf6ea058a387faee0a9f33c8d25c6ba05e1be`
- License source: https://raw.githubusercontent.com/adobe-fonts/source-serif/80d3f8894c09c937bebfa9011247d2e1c79fd6f4/LICENSE.md
- Original local WOFF2 SHA256: `940a76eda1388de39d38c8e7a79bf6ea058a387faee0a9f33c8d25c6ba05e1be`
- Variation axes: `wght` 200–900, default 400; `opsz` 8–60, default 20
- Original import processing: Unmodified official WOFF2 file; filename only changed where needed.

## Original upstream Inter Variable

- Repository: https://github.com/rsms/inter
- Fixed commit: `353b61b9f4430d5f420d56605a6e7993e0941470`
- Official source: https://raw.githubusercontent.com/rsms/inter/353b61b9f4430d5f420d56605a6e7993e0941470/docs/font-files/InterVariable.woff2
- Original source SHA256: `693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3`
- License source: https://raw.githubusercontent.com/rsms/inter/353b61b9f4430d5f420d56605a6e7993e0941470/LICENSE.txt
- Original local WOFF2 SHA256: `693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3`
- Variation axes: `opsz` 14–32, default 14; `wght` 100–900, default 400
- Original import processing: Unmodified official WOFF2 file; filename only changed where needed.

## Original upstream JetBrains Mono

- Repository: https://github.com/JetBrains/JetBrainsMono
- Fixed commit: `19371302b95d218af43299bce79ddbddd0bc364d`
- Official source: https://raw.githubusercontent.com/JetBrains/JetBrainsMono/19371302b95d218af43299bce79ddbddd0bc364d/fonts/variable/JetBrainsMono%5Bwght%5D.ttf
- Original source SHA256: `3cfafa86e28b87184d592fef82846e8c10cb48653c62efcda34f082da225ec34`
- License source: https://raw.githubusercontent.com/JetBrains/JetBrainsMono/19371302b95d218af43299bce79ddbddd0bc364d/OFL.txt
- Original local WOFF2 SHA256: `2351228fa6d146e978e6a2dea44aab054d7e472545f97fe5ed99726901ff38b7`
- Variation axes: `wght` 100–800, default 400
- Original import processing: Official upstream variable TTF converted to WOFF2 with fontTools 4.63.0, recalcTimestamp=False; glyph design, internal font names and variation axes retained.
