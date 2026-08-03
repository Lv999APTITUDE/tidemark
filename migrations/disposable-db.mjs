// 一次性随机库的生命周期封装（Codex P0-05 round-6：CREATE、首次 connect、fn 必须
// 同处最外层 try——首次连接失败也要走 cleanup，否则 CREATE 成功+connect 风暴=库残留）。
// connect 可注入：单测用 fake 断言"connect 抛错仍恰好 DROP 一次、且只 DROP 本次建的库"。
import { quoteIdentifier, withDatabase } from './db.mjs'

export const withDisposableDb = async ({ base, mkName, connect, fn, log = console }) => {
  const DB = mkName()
  let created = false
  let client = null
  try {
    const admin = await connect(base, { label: 'admin' })
    try {
      await admin.query(`CREATE DATABASE ${quoteIdentifier(DB)}`)   // 无 IF NOT EXISTS：撞名 fail-closed
      created = true
    } finally { await admin.end().catch(() => {}) }
    client = await connect(withDatabase(base, DB), { label: DB })
    // act 内查询的断线重连壳（跨国线路会随机掐已建连接）：首连保持即时（unit 契约不变），
    // 后续 query 撞瞬断则换新连接重试该条——act 语句无显式事务，单句重试安全
    const TRANSIENT = new Set(['ECONNRESET', 'ETIMEDOUT', 'EPIPE', '57P01', '08006', '08001'])
    const resilient = {
      query: async (text, values) => {
        for (let i = 1; ; i++) {
          try { return await client.query(text, values) }
          catch (e) {
            const transient = TRANSIENT.has(e.code) || /Connection terminated|read ECONNRESET/.test(e.message ?? '')
            if (!transient || i >= 5) throw e
            log.error?.(`[retry] disposable query ${i}/5 (${e.code ?? 'conn'}); reconnecting`)
            try { await client.end() } catch {}
            await new Promise(r => setTimeout(r, 500 * i))
            client = await connect(withDatabase(base, DB), { label: DB })
          }
        }
      },
      end: () => client.end(),
    }
    return await fn(resilient, DB)   // 库名作第二参（可选消费）：子进程类步骤（如 backfill）需要它
  } finally {
    await client?.end().catch(() => {})
    if (created) {
      try {
        const admin2 = await connect(base, { label: 'admin-cleanup' })
        try {
          await admin2.query(`DROP DATABASE ${quoteIdentifier(DB)} CASCADE`)
          log.log(`disposable db dropped: ${DB}`)
        } finally { await admin2.end().catch(() => {}) }
      } catch (e) {
        // 不吞主错误：报告残留并置退出码，人工按库名清理
        log.error(`CLEANUP FAILED: disposable db ${DB} still exists: ${e.message}`)
        process.exitCode = 1
      }
    }
  }
}
