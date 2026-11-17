# Self hosted open source fonts

Downloaded on 2026-11-16 from the official upstream repositories listed below. Font files are used for Latin reading text, interface labels and code. Chinese text should use system fallbacks; these files do not supply a CJK font. No Anthropic proprietary fonts or logos are included.

All three fonts use the SIL Open Font License 1.1. Their copyright and license notices are retained in each font directory as OFL.txt. Internal font family names have not been changed. Retain these notices when redistributing the font files.

| Local file | Internal typographic family | Normal style weight range | Bytes |
| --- | --- | --- | --- |
| source-serif-4/SourceSerif4Variable-Roman.woff2 | Source Serif 4 Variable | 200–900 | 429,100 |
| inter/InterVariable.woff2 | Inter Variable | 100–900 | 352,240 |
| jetbrains-mono/JetBrainsMonoVariable.woff2 | JetBrains Mono | 100–800 | 113,704 |

Total font payload: 895,044 bytes, excluding license text. These are normal/Roman fonts; no italic font has been added.

## Source Serif 4 Variable

- Repository: https://github.com/adobe-fonts/source-serif
- Fixed commit: `80d3f8894c09c937bebfa9011247d2e1c79fd6f4`
- Official source: https://raw.githubusercontent.com/adobe-fonts/source-serif/80d3f8894c09c937bebfa9011247d2e1c79fd6f4/WOFF2/VAR/SourceSerif4Variable-Roman.ttf.woff2
- Original source SHA256: `940a76eda1388de39d38c8e7a79bf6ea058a387faee0a9f33c8d25c6ba05e1be`
- License source: https://raw.githubusercontent.com/adobe-fonts/source-serif/80d3f8894c09c937bebfa9011247d2e1c79fd6f4/LICENSE.md
- Local WOFF2 SHA256: `940a76eda1388de39d38c8e7a79bf6ea058a387faee0a9f33c8d25c6ba05e1be`
- Variation axes: `wght` 200–900, default 400; `opsz` 8–60, default 20
- Processing: Unmodified official WOFF2 file; filename only changed where needed.

## Inter Variable

- Repository: https://github.com/rsms/inter
- Fixed commit: `353b61b9f4430d5f420d56605a6e7993e0941470`
- Official source: https://raw.githubusercontent.com/rsms/inter/353b61b9f4430d5f420d56605a6e7993e0941470/docs/font-files/InterVariable.woff2
- Original source SHA256: `693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3`
- License source: https://raw.githubusercontent.com/rsms/inter/353b61b9f4430d5f420d56605a6e7993e0941470/LICENSE.txt
- Local WOFF2 SHA256: `693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3`
- Variation axes: `opsz` 14–32, default 14; `wght` 100–900, default 400
- Processing: Unmodified official WOFF2 file; filename only changed where needed.

## JetBrains Mono

- Repository: https://github.com/JetBrains/JetBrainsMono
- Fixed commit: `19371302b95d218af43299bce79ddbddd0bc364d`
- Official source: https://raw.githubusercontent.com/JetBrains/JetBrainsMono/19371302b95d218af43299bce79ddbddd0bc364d/fonts/variable/JetBrainsMono%5Bwght%5D.ttf
- Original source SHA256: `3cfafa86e28b87184d592fef82846e8c10cb48653c62efcda34f082da225ec34`
- License source: https://raw.githubusercontent.com/JetBrains/JetBrainsMono/19371302b95d218af43299bce79ddbddd0bc364d/OFL.txt
- Local WOFF2 SHA256: `2351228fa6d146e978e6a2dea44aab054d7e472545f97fe5ed99726901ff38b7`
- Variation axes: `wght` 100–800, default 400
- Processing: Official upstream variable TTF converted to WOFF2 with fontTools 4.63.0, recalcTimestamp=False; glyph design, internal font names and variation axes retained.
