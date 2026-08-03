// local-onnx embedding 实现（结论 55 主路径）：Lambda/本地进程内自托管推理，零外部调用。
// - 模型完整性：首次使用前按 embed-manifest.json 逐文件验 SHA256（缺/错即抛，fail-closed）
// - 单例：pipeline 是 module 级 Promise，并发共享一次加载
// - 零出网：allowRemoteModels=false + localModelPath=<repo>/models（TIDEMARK_MODEL_DIR 可覆盖，测试用）
// - 推理路径 = transformers.js FeatureExtractionPipeline（mean pooling + L2 normalize），
//   与 spike 完全同源——等价性由 test-embed-onnx 的 spike digest 锚点钉死。不手写池化：
//   浮点累加精度次序不同就会破坏跨平台 bit 级一致性（实测教训）。
// - 截断契约（六条边界 #5 冻结值；round-3 P1-2 修正 off-by-specials）：max_tokens=**256**
//   是【送进模型的最终 token 数上限，含 [CLS]/[SEP]】，由 manifest.output 冻结并进入派生身份。
//   实现 = head-content-tokens-v2：无特殊符编码取内容 token -> 预算 = max_tokens - 特殊符数
//  （实测 2）-> 取前 254 个内容 token -> decode -> 送 pipeline（短文本与 spike 逐字节同源）。
//   【硬断言】截断文本重编码（含特殊符）必须 <= max_tokens，违约直接抛——"喂给模型 257 个"
//   这类边界溢出永远不可能静默发生。确定性：同内容前缀 -> 同 decode 文本 -> 同向量
//  （E5b/E5c 以首个被排除 token 为判别子）。可观测：truncated/token_count/model_input_tokens。
// - 输出 384 维后零填充 512（cosine/L2 精确保持，VECTOR(512) schema 不动）
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { embedIdentity } from './embed-identity.mjs'

let loaded = null
const load = async () => {
  const { manifest, repo_root, embedding_model_id } = embedIdentity()
  const modelDir = process.env.TIDEMARK_MODEL_DIR || join(repo_root, 'models')
  const base = join(modelDir, manifest.model_repo)
  for (const [rel, expected] of Object.entries(manifest.files)) {
    let buf
    try { buf = readFileSync(join(base, rel)) }
    catch { throw new Error(`model artifact missing: ${rel} (fail-closed; run infra/fetch-model.mjs)`) }
    const actual = createHash('sha256').update(buf).digest('hex')
    if (actual !== expected) throw new Error(`model artifact SHA mismatch: ${rel} expected ${expected} got ${actual}`)
  }
  const { pipeline, env } = await import('@huggingface/transformers')
  env.allowRemoteModels = false
  env.localModelPath = modelDir
  const extractor = await pipeline('feature-extraction', manifest.model_repo, { dtype: manifest.output.dtype })
  const maxTokens = manifest.output.max_tokens   // 冻结的策略值（进身份）：模型最终输入上限，含特殊符
  if (!Number.isInteger(maxTokens) || maxTokens <= 0) throw new Error(`manifest output.max_tokens invalid (${maxTokens})`)
  if (manifest.output.truncation !== 'head-content-tokens-v2') throw new Error(`unknown truncation policy ${manifest.output.truncation}`)
  return { extractor, manifest, embedding_model_id, maxTokens }
}

export const embedLocalOnnx = async (text) => {
  loaded ??= load()
  const { extractor, manifest, embedding_model_id, maxTokens } = await loaded
  // head-content-tokens-v2：全文计数（含特殊符，=模型输入真相）可观测；超界时按
  // 内容 token 预算（max - 特殊符数）截断，decode 后重编码【硬断言】不超上限
  const tk = extractor.tokenizer
  const fullIds = tk.encode(text)
  const tokenCount = fullIds.length
  const truncated = tokenCount > maxTokens
  let input = text
  let modelInputTokens = tokenCount
  if (truncated) {
    const contentIds = tk.encode(text, { add_special_tokens: false })
    const specials = fullIds.length - contentIds.length
    input = tk.decode(contentIds.slice(0, maxTokens - specials), { skip_special_tokens: true })
    modelInputTokens = tk.encode(input).length
    if (modelInputTokens > maxTokens) {
      throw new Error(`truncation contract violated: model input would be ${modelInputTokens} > ${maxTokens} tokens (fail-closed)`)
    }
    console.log(JSON.stringify({ evt: 'embed_truncated', token_count: tokenCount, model_input_tokens: modelInputTokens, max_tokens: maxTokens, chars: text.length }))
  }
  const out = await extractor(input, { pooling: manifest.output.pooling, normalize: manifest.output.normalize })
  const v = Array.from(out.data)
  if (v.length !== manifest.output.native_dims) throw new Error(`expected ${manifest.output.native_dims} dims, got ${v.length}`)
  const padded = [...v, ...new Array(manifest.output.pad_to - manifest.output.native_dims).fill(0)]
  return { vector: padded, embedding_model_id, truncated, token_count: tokenCount, model_input_tokens: modelInputTokens }
}
