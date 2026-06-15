# Compatible Clients

Local-multimodal-mcp works with any client that implements the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP).

## OpenCode (Primary Target)

[OpenCode](https://opencode.ai) is the primary use case for this project.
It gives vision and image generation to text-only models like **DeepSeek v4**,
which cannot natively process images.

```json
{
  "mcpServers": {
    "blind-vision-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/blind-vision-mcp", "blind-vision-mcp"]
    }
  }
}
```

## Claude Desktop

[Claude Desktop](https://claude.ai/desktop) supports MCP natively via
`claude_desktop_config.json`.

## Cursor

[Cursor](https://cursor.sh) supports MCP in its settings panel.
Add a new MCP server pointing to the blind-vision-mcp command.

## Cline

[Cline](https://github.com/cline/cline) supports MCP servers in its
configuration.

## Any MCP Client

The server uses **stdio transport** and the standard MCP protocol.
Any MCP-compatible client can connect to it.
