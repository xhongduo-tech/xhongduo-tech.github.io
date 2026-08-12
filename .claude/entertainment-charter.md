# 娱乐大典 · 内容宪章（Agent 必读）

这是为娱乐页「综合大典版」生成内容时给每位 Agent 的规范。

## 条目 Schema（四类一致）

```json
{
  "id": "blade-runner-2049",
  "title": "银翼杀手 2049",
  "en": "Blade Runner 2049",
  "year": 2017,
  "genres": [
    "scifi",
    "noir"
  ],
  "creator": "Denis Villeneuve",
  "region": "US",
  "rating": {
    "douban": 8.3,
    "imdb": 8,
    "metacritic": 81
  },
  "awards": [
    "oscar"
  ],
  "note": "关于「什么是人」的视觉诗。",
  "noteEn": "A visual poem on what makes us human."
}
```

- 必填：`title`（中文名）、`en`（英文名，去重键）、`year`（整数，范围见下）、`genres`（1–3 个本类流派键）、`creator`（导演/工作室/艺术家/摄影师）、`region`（ISO 代码）、`note`（一句中文打动理由）、`noteEn`（一句英文理由）。
- 可空：`rating`（对象，键在本类评分源内、值在刻度内）、`awards`（本类奖项键数组）。`id` 缺省由 en 派生，可省略。
- 质量红线：**禁止编造**。年份/创作者/奖项必须真实可核实；存疑事实最多用 2 次 WebSearch 核实。`note`/`noteEn` 要具体、非套话（说清它为何好/独特，而非「经典之作」）。

## movies（电影）· year 1880–2027

- 流派键：drama(🎭剧情/Drama) comedy(😂喜剧/Comedy) action(💥动作/Action) scifi(🛸科幻/Sci-Fi) fantasy(🐉奇幻/Fantasy) horror(👻恐怖/Horror) thriller(🕵️悬疑/Thriller) crime(🔫犯罪/Crime) romance(❤️爱情/Romance) animation(🎨动画/Animation) documentary(🎥纪录/Documentary) war(⚔️战争/War) western(🤠西部/Western) musical(🎶歌舞/Musical) biopic(🎬传记/Biopic) noir(🕶️黑色/Noir)
- 奖项键：oscar(奥斯卡) palme-dor(戛纳金棕榈) golden-lion(威尼斯金狮) golden-bear(柏林金熊) bafta(英国学院奖) golden-globe(金球奖) cesar(凯撒奖) golden-rooster(金鸡奖) golden-horse(金马奖) hk-film-award(金像奖) blue-dragon(青龙奖) japan-academy(日本学院奖) siff(金爵奖) sundance(圣丹斯评审团)
- 评分源：douban(0–10) imdb(0–10) metacritic(0–100)

## games（游戏）· year 1958–2027

- 流派键：action-adventure(⚔️动作冒险/Action-Adventure) rpg(🧙角色扮演/RPG) open-world(🌍开放世界/Open World) shooter(🔫射击/Shooter) platformer(🏃平台跳跃/Platformer) puzzle(🧩解谜/Puzzle) strategy(♟️策略/Strategy) sim(🛠️模拟/Simulation) indie(🎲独立/Indie) horror(👻恐怖/Horror) sports(🏅体育/Sports) racing(🏎️竞速/Racing) fighting(🥊格斗/Fighting) rhythm(🎵节奏/Rhythm) visual-novel(📖视觉小说/Visual Novel) party(🎉派对/Party)
- 奖项键：tga-goty(年度最佳游戏) dice(DICE 年度游戏) bafta-games(BAFTA 游戏奖) golden-joystick(金摇杆) japan-game-award(日本游戏大赏) gdc(GDC 开发者选择) igf(独立游戏节)
- 评分源：metacritic(0–100) steam(0–10) ign(0–10)

## music（音乐）· year 1400–2027

- 流派键：classical(🎻古典/Classical) jazz(🎷爵士/Jazz) rock(🎸摇滚/Rock) pop(🎤流行/Pop) electronic(🎛️电子/Electronic) folk-country(🪕民谣乡村/Folk & Country) hiphop(🎧嘻哈/Hip-Hop) rnb-soul(🎙️节奏布鲁斯/R&B / Soul) blues(🎹蓝调/Blues) world(🌍世界音乐/World) chinese-pop(🎶华语流行/Chinese Pop) japanese-korean(🎌日韩音乐/JP/KR Music) soundtrack(🎬影视原声/Soundtrack) ambient-newage(🌙氛围/新世纪/Ambient & New Age)
- 奖项键：grammy(格莱美) mercury(水星奖) oscar-score(奥斯卡配乐) golden-melody(金曲奖) brit-award(全英音乐奖) polar-music(极地音乐奖)
- 评分源：rym(0–10) pitchfork(0–10) douban(0–10)

## photography（摄影）· year 1800–2027

- 流派键：street(📸街头/Street) bw(⬛黑白/Black & White) landscape(🏔️风光/Landscape) portrait(👤人像/Portrait) documentary(🗞️纪实/Documentary) minimal(⬜极简/Minimalist) architecture(🏛️建筑/Architecture) fashion(👗时尚/Fashion) nature-wildlife(🦁自然生态/Nature & Wildlife) astro(🌌天文/Astro) night(🌃夜景/Night) urban(🏙️城市/Urban)
- 奖项键：world-press-photo(荷赛奖) pulitzer(普利策) hasselblad(哈苏奖) leica-oscar-barnack(徕卡巴纳克奖) sony-world-photo(索尼世界摄影奖) nat-geo(国家地理摄影师) taylor-wessing(泰勒·韦辛肖像奖)
- 评分源：无（本类无评分）

## 地区键（ISO）

CN HK TW US UK FR DE IT ES JP KR RU IN BR CA AU SE NO NL CH IE PL CZ AT BE PT GR TR IL TH SG MY ID NZ ZA EG MX AR CL CO VE UA RO DK FI IS LU MC SA AE NG KE

## 红线

- 只写你自己那份 raw JSON 文件，输出必须是合法 JSON **数组**（2 空格缩进）。
- 不跑 git、不碰其他文件/其他分类的 raw 文件、不运行合并脚本、不开子 Agent。
- 同名作品（翻拍/重制/重制版）用可区分的 en 标题，避免去重冲突。
- 完成后返回简短报告：写入条数、各流派计数、你做的消歧决定。
