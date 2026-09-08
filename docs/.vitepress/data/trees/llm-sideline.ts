import { fromOutline, type Outline } from './schema'

/** 附录支线：型号、产品、世界模型。光刻已独立成栏。不插入主干 prev/next。 */
const extra: Outline[] = [
  [
    '世界模型与空间智能',
    [
      [
        '李飞飞 / World Labs',
        [
          [
            '空间智能纲领',
            [
              '空间智能：感知、推理、在三维中行动|spatial-intelligence',
              '世界模型四件事：重建、生成、仿真、交互|world-model-four-roles',
              '持久三维世界 vs 边走边生成帧|persistent-3d-vs-streaming-frames',
            ],
          ],
          [
            'Marble 生成式世界',
            [
              '多模态提示到三维世界（文/图/视频/布局）|marble-multimodal-prompt',
              '多图与视频的视角拼接成一致场景|marble-multi-view-stitch',
              'Chisel：粗几何定结构、文本定风格|marble-chisel',
              '区域扩展与 Composer 拼世界大图|marble-expand-compose',
              '三维高斯溅射作为高保真表示|marble-gaussian-splats',
              '碰撞网格与视觉网格双导出|marble-dual-mesh',
              'Spark：浏览器高斯溅射渲染|marble-spark',
              '结构保持的视频增强与动态元素|marble-video-enhance',
              'AI 原生局部编辑与风格改写|marble-world-edit',
            ],
          ],
          [
            'RTFM 实时帧模型',
            [
              'RTFM：探索时实时出帧而非导出场景|worldlabs-rtfm',
              '实时世界模型的形变与不一致性|rtfm-morphing',
            ],
          ],
          [
            'Atlas Omni 世界模型',
            [
              '多模态自回归扩散 Transformer|atlas-ardt',
              '共享空间上下文：图像锚定在三维位姿|atlas-spatial-context',
              '相机位姿作为原生输入而非文本描述|atlas-native-camera',
              '视频作为带位姿的图像序列|atlas-video-as-frames',
              'Rectified flow 潜空间扩散|atlas-rectified-flow',
              '深度图、点云与高斯溅射写出|atlas-3d-writeout',
              '稀疏视角新视角合成与三维重建|atlas-sparse-view-recon',
              '相机可控长视频（至 1440p / 1 分钟）|atlas-camera-controlled-video',
              '多机位 reframing 与子弹时间|atlas-video-reframe',
              'Real-to-Sim：重建场景并生成机器人传感器视图|atlas-real-to-sim',
              '沿用 LLM 的 KV cache 与分离式服务|atlas-llm-serving-tricks',
              '扩散蒸馏、CFG 与 VAE 潜空间|atlas-diffusion-stack',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '模型族',
    [
      [
        'Dense 开源',
        [
          [
            'Llama 系',
            [
              'Llama 1 架构选择|llama-1',
              'Llama 2 GQA 与对话|llama-2',
              'Llama 3 数据与 tokenizer|llama-3',
              'Llama 3.1 长上下文|llama-3-1',
              'Llama 4 与 MoE 方向|llama-4',
              'Code Llama|code-llama',
            ],
          ],
          [
            'Qwen / GLM / Gemma / Mistral',
            [
              'Qwen 1.5 / 2 / 2.5 演进|qwen-evolution',
              'Qwen3|qwen3',
              'Qwen2-Audio / Qwen2.5-Omni 语音|qwen-audio-omni',
              'Qwen3-ASR-1.7B / 0.6B|qwen3-asr',
              'Qwen3-VL 与文档 OCR|qwen3-vl',
              'Qwen3.5-OCR|qwen35-ocr-model',
              'GLM 与 ChatGLM|glm',
              'GLM-4|glm-4',
              'Gemma / Gemma 2|gemma',
              'Mistral 与 Mixtral|mistral-mixtral',
              'Phi 小模型路线|phi',
            ],
          ],
        ],
      ],
      [
        'MoE 与推理导向',
        [
          [
            'DeepSeek 系',
            [
              'DeepSeek-V2 MLA 与 MoE|deepseek-v2',
              'DeepSeek-V3|deepseek-v3',
              'DeepSeek-R1 与推理时行为|deepseek-r1',
              'DeepSeek 开源栈与部署约束|deepseek-serving',
            ],
          ],
          [
            '其他 MoE',
            [
              'Mixtral 8x7B / 8x22B|mixtral',
              'DBRX|dbrx',
              'Grok MoE 公开信息|grok-moe',
              'OLMoE|olmoe',
            ],
          ],
        ],
      ],
      [
        '小模型与端侧',
        [
          [
            '结构与蒸馏',
            [
              'SLM 的能力边界|slm-capability',
              '端侧上下文与 KV 预算|on-device-kv',
              'NPU 友好算子|npu-friendly-ops',
              '蒸馏到端侧的数据与温度|on-device-distill',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    'Qwen 文档与语音产品',
    [
      [
        '文档解析',
        [
          [
            'Qwen OCR 与文档解析',
            [
              'Naive Dynamic Resolution 原生分辨率切块|qwen-vl-naive-dynamic-res',
              '2×2 patch merge 控制视觉 token 数|qwen-vl-patch-merge',
              '窗口注意力与周期性全局注意力交替|qwen-vl-window-full-attn',
              'MRoPE / Interleaved MRoPE 时空位置|qwen3-vl-interleaved-mrope',
              'DeepStack：多层 ViT 特征注入 LLM|qwen3-vl-deepstack',
              'SigLIP-2 视觉骨干|qwen3-vl-siglip2',
              'Qwen HTML 版式感知文档解析|qwen-html-document-parse',
              '文字定位与 2D grounding|qwen-ocr-text-grounding',
              '表格、公式与卡证关键信息抽取|qwen-ocr-kie',
              '粗到细伪标注 OCR 数据管线|qwen-ocr-coarse-to-fine',
              '多页 PDF 合成与跨页文档 VQA|qwen-ocr-long-pdf',
              '图像旋转矫正|qwen-ocr-rotation',
              'Qwen-VL-OCR 内置任务模板|qwen-vl-ocr-tasks',
              'Qwen3.5-OCR：原生 PDF 与多轮抽取|qwen35-ocr',
            ],
          ],
        ],
      ],
      [
        '语音转写',
        [
          [
            'Qwen ASR',
            [
              'LALM：先理解音频再生成转写|qwen3-asr-lalm',
              'Qwen3-Omni 作为语音理解基座|qwen3-omni-speech-base',
              'AuT：AED 音频 Transformer 编码器|qwen3-asr-aut',
              '128 维 Fbank 与 Conv2D 8× 下采样|qwen3-asr-fbank-downsample',
              '12.5 Hz 音频 token 率|qwen3-asr-token-rate',
              '动态 FlashAttention 窗口 1s–8s|qwen3-asr-dynamic-window',
              '分块 Conv2D（约 100 帧 → 13 token）|qwen3-asr-chunked-conv',
              '学习型 projector 对齐 AuT 与 Qwen3|qwen3-asr-projector',
              'Qwen3 解码器：GQA、RoPE、QK-Norm|qwen3-asr-decoder',
              '流式与离线统一推理|qwen3-asr-streaming-offline',
              '语言识别与 52 语种/方言|qwen3-asr-lid',
              'Qwen3-ForcedAligner 非自回归时间戳|qwen3-forced-aligner',
              '伪标注大规模语音预训练|qwen3-asr-pseudo-label',
              'vLLM 批推理与流式 ASR 服务|qwen3-asr-vllm',
            ],
          ],
        ],
      ],
    ],
  ],
]

export const llmSideline = fromOutline(extra)
