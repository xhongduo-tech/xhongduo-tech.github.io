import type { UserConfig } from 'vitepress'
import type { Plugin } from 'vite'
import { withMermaid } from 'vitepress-plugin-mermaid'

/**
 * Keep the upstream Markdown renderer, virtual settings and compatibility aliases.
 * The theme registers the upstream component asynchronously, so the plugin must
 * not also import it statically into VitePress's global application entry.
 */
export function withLazyMermaid(config: UserConfig): UserConfig {
  const resolved = withMermaid(config)
  let removedGlobalRegistration = false

  resolved.vite!.plugins = resolved.vite!.plugins!.map((option) => {
    const plugin = option as Plugin | undefined
    if (plugin?.name !== 'vite-plugin-mermaid') return option

    const { transform: _globalRegistration, ...virtualSettingsPlugin } = plugin
    removedGlobalRegistration = true
    return virtualSettingsPlugin
  })

  if (!removedGlobalRegistration) {
    throw new Error('Mermaid integration changed: review its global component registration before upgrading.')
  }

  return resolved
}
